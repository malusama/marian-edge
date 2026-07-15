use std::mem::size_of;

use crate::metal_runtime::{Buffer, Commands, MetalParams, MetalStorage, grid};

use super::{MetalEngine, checked_mul, checked_product, to_u32};

const MAXIMUM_SCORE_BYTES: usize = 1_024 * 1_024 * 1_024;
const MATMUL_TILE: usize = 32;
const MATMUL_THREADS_Y: usize = 8;
const FLASH_ATTENTION_THREADS: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
struct MatMulParams {
    rows: u32,
    cols: u32,
    inner: u32,
    has_bias: u32,
    activation: u32,
    storage: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbeddingParams {
    batch: u32,
    sequence: u32,
    dim: u32,
    storage: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DecoderInputParams {
    batch: u32,
    dim: u32,
    position: u32,
    storage: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NormParams {
    rows: u32,
    dim: u32,
    storage: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttentionParams {
    batch: u32,
    query_length: u32,
    key_length: u32,
    dim: u32,
    heads: u32,
    query_stride: u32,
    query_offset: u32,
    key_stride: u32,
    key_offset: u32,
    value_stride: u32,
    value_offset: u32,
    query_tile: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SelectDecodeParams {
    batch: u32,
    candidates: u32,
    dim: u32,
    storage: u32,
    step: u32,
    history_step: u32,
    threads: u32,
}

// SAFETY: Every parameter is repr(C), contains only u32 fields, and mirrors
// the structure with the same name in kernels.metal.
unsafe impl MetalParams for MatMulParams {}
unsafe impl MetalParams for EmbeddingParams {}
unsafe impl MetalParams for DecoderInputParams {}
unsafe impl MetalParams for NormParams {}
unsafe impl MetalParams for AttentionParams {}
unsafe impl MetalParams for SelectDecodeParams {}

#[derive(Clone, Copy)]
pub(super) struct AttentionView<'a> {
    pub(super) buffer: &'a Buffer,
    pub(super) stride: usize,
    pub(super) offset: usize,
}

impl<'a> AttentionView<'a> {
    pub(super) const fn contiguous(buffer: &'a Buffer, dim: usize) -> Self {
        Self {
            buffer,
            stride: dim,
            offset: 0,
        }
    }
}

impl MetalEngine {
    pub(super) fn embedding(
        &self,
        commands: &Commands<'_>,
        tokens: &Buffer,
        batch: usize,
        sequence: usize,
    ) -> Result<Buffer, String> {
        let output = self.f32_buffer(checked_product(&[batch, sequence, self.dim])?)?;
        let params = EmbeddingParams {
            batch: to_u32(batch, "batch")?,
            sequence: to_u32(sequence, "sequence")?,
            dim: to_u32(self.dim, "model dimension")?,
            storage: self.storage.code(),
        };
        commands.dispatch(
            &commands.runtime().pipelines.embedding,
            &[
                tokens,
                &self.model.encoder_embedding,
                &self.positions,
                &output,
            ],
            &params,
            grid(self.dim, sequence, batch),
            grid(16, 8, 1),
        )?;
        Ok(output)
    }

    pub(super) fn decoder_input(
        &self,
        commands: &Commands<'_>,
        previous: &Buffer,
        batch: usize,
        position: usize,
    ) -> Result<Buffer, String> {
        let output = self.f32_buffer(checked_mul(batch, self.dim)?)?;
        let params = DecoderInputParams {
            batch: to_u32(batch, "batch")?,
            dim: to_u32(self.dim, "model dimension")?,
            position: to_u32(position, "decoder position")?,
            storage: self.storage.code(),
        };
        commands.dispatch(
            &commands.runtime().pipelines.decoder_input,
            &[
                previous,
                &self.model.decoder_embedding,
                &self.positions,
                &output,
            ],
            &params,
            grid(self.dim, batch, 1),
            grid(32, 4, 1),
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul(
        &self,
        commands: &Commands<'_>,
        lhs: &Buffer,
        rhs: &Buffer,
        bias: Option<&Buffer>,
        rows: usize,
        cols: usize,
        inner: usize,
    ) -> Result<Buffer, String> {
        self.matmul_with_activation(commands, lhs, rhs, bias, rows, cols, inner, false, false)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul_persistent(
        &self,
        commands: &Commands<'_>,
        lhs: &Buffer,
        rhs: &Buffer,
        bias: Option<&Buffer>,
        rows: usize,
        cols: usize,
        inner: usize,
    ) -> Result<Buffer, String> {
        self.matmul_with_activation(commands, lhs, rhs, bias, rows, cols, inner, false, true)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul_with_activation(
        &self,
        commands: &Commands<'_>,
        lhs: &Buffer,
        rhs: &Buffer,
        bias: Option<&Buffer>,
        rows: usize,
        cols: usize,
        inner: usize,
        relu: bool,
        persistent: bool,
    ) -> Result<Buffer, String> {
        require_f32(lhs, checked_mul(rows, inner)?, "matrix lhs")?;
        require_model(rhs, checked_mul(inner, cols)?, self.storage, "matrix rhs")?;
        if let Some(bias) = bias {
            require_model(bias, cols, self.storage, "matrix bias")?;
        }
        let elements = checked_mul(rows, cols)?;
        let output = if persistent {
            self.workspace.request_f32(&self.runtime, elements)?
        } else {
            self.f32_buffer(elements)?
        };
        let params = MatMulParams {
            rows: to_u32(rows, "matrix rows")?,
            cols: to_u32(cols, "matrix columns")?,
            inner: to_u32(inner, "matrix inner dimension")?,
            has_bias: u32::from(bias.is_some()),
            activation: u32::from(relu),
            storage: self.storage.code(),
        };
        if self.storage == MetalStorage::Fp32 && !self.tuning.gemm.use_custom_fp32(rows, cols) {
            commands.mps_matmul(lhs, rhs, &output, rows, cols, inner)?;
            if bias.is_some() || relu {
                commands.dispatch(
                    &commands.runtime().pipelines.matmul_bias_activation,
                    &[&output, bias.unwrap_or(&self.dummy_bias)],
                    &params,
                    grid(cols, rows, 1),
                    grid(32, 8, 1),
                )?;
            }
        } else {
            commands.dispatch_threadgroups(
                &commands.runtime().pipelines.matmul_microtile,
                &[lhs, rhs, bias.unwrap_or(&self.dummy_bias), &output],
                &params,
                grid(cols.div_ceil(MATMUL_TILE), rows.div_ceil(MATMUL_TILE), 1),
                grid(MATMUL_TILE, MATMUL_THREADS_Y, 1),
            )?;
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn linear_residual_norm(
        &self,
        commands: &Commands<'_>,
        input: &Buffer,
        weights: &Buffer,
        linear_bias: &Buffer,
        residual: &Buffer,
        scale: &Buffer,
        norm_bias: &Buffer,
        rows: usize,
        inner: usize,
    ) -> Result<Buffer, String> {
        let elements = checked_mul(rows, self.dim)?;
        let matrix_product = self.matmul(commands, input, weights, None, rows, self.dim, inner)?;
        require_f32(&matrix_product, elements, "linear projection")?;
        require_model(
            linear_bias,
            self.dim,
            self.storage,
            "linear projection bias",
        )?;
        require_f32(residual, elements, "layer norm residual")?;
        require_model(scale, self.dim, self.storage, "layer norm scale")?;
        require_model(norm_bias, self.dim, self.storage, "layer norm bias")?;
        let output = self.f32_buffer(elements)?;
        let params = NormParams {
            rows: to_u32(rows, "layer norm rows")?,
            dim: to_u32(self.dim, "model dimension")?,
            storage: self.storage.code(),
        };
        commands.dispatch_threadgroups(
            &commands.runtime().pipelines.bias_residual_norm,
            &[
                &matrix_product,
                linear_bias,
                residual,
                scale,
                norm_bias,
                &output,
            ],
            &params,
            grid(rows, 1, 1),
            grid(128, 1, 1),
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ssru_norm(
        &self,
        commands: &Commands<'_>,
        projection: &Buffer,
        forget_bias: &Buffer,
        state: &Buffer,
        residual: &Buffer,
        scale: &Buffer,
        bias: &Buffer,
        rows: usize,
    ) -> Result<Buffer, String> {
        let elements = checked_mul(rows, self.dim)?;
        require_f32(
            projection,
            checked_mul(elements, 2)?,
            "SSRU packed projection",
        )?;
        for (buffer, label) in [(state, "SSRU state"), (residual, "SSRU residual")] {
            require_f32(buffer, elements, label)?;
        }
        require_model(forget_bias, self.dim, self.storage, "SSRU forget bias")?;
        require_model(scale, self.dim, self.storage, "SSRU scale")?;
        require_model(bias, self.dim, self.storage, "SSRU bias")?;
        let output = self.f32_buffer(elements)?;
        let params = NormParams {
            rows: to_u32(rows, "SSRU rows")?,
            dim: to_u32(self.dim, "model dimension")?,
            storage: self.storage.code(),
        };
        commands.dispatch_threadgroups(
            &commands.runtime().pipelines.ssru_norm,
            &[
                projection,
                forget_bias,
                state,
                residual,
                scale,
                bias,
                &output,
            ],
            &params,
            grid(rows, 1, 1),
            grid(128, 1, 1),
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attend(
        &self,
        commands: &Commands<'_>,
        query: AttentionView<'_>,
        key: AttentionView<'_>,
        value: AttentionView<'_>,
        lengths: &Buffer,
        batch: usize,
        query_length: usize,
        key_length: usize,
    ) -> Result<Buffer, String> {
        let head_dim = self.dim / self.heads;
        if self
            .tuning
            .attention
            .use_flash(query_length, key_length, head_dim)
        {
            let output = self.f32_buffer(checked_product(&[batch, query_length, self.dim])?)?;
            let params = AttentionParams {
                batch: to_u32(batch, "attention batch")?,
                query_length: to_u32(query_length, "attention query length")?,
                key_length: to_u32(key_length, "attention key length")?,
                dim: to_u32(self.dim, "model dimension")?,
                heads: to_u32(self.heads, "attention heads")?,
                query_stride: to_u32(query.stride, "attention query stride")?,
                query_offset: to_u32(query.offset, "attention query offset")?,
                key_stride: to_u32(key.stride, "attention key stride")?,
                key_offset: to_u32(key.offset, "attention key offset")?,
                value_stride: to_u32(value.stride, "attention value stride")?,
                value_offset: to_u32(value.offset, "attention value offset")?,
                query_tile: to_u32(self.tuning.attention.query_tile(), "attention query tile")?,
            };
            commands.dispatch_threadgroups(
                &commands.runtime().pipelines.attention_flash,
                &[query.buffer, key.buffer, lengths, value.buffer, &output],
                &params,
                grid(
                    query_length.div_ceil(self.tuning.attention.query_tile()),
                    batch * self.heads,
                    1,
                ),
                grid(FLASH_ATTENTION_THREADS, 1, 1),
            )?;
            return Ok(output);
        }
        let score_elements = checked_product(&[batch, self.heads, query_length, key_length])?;
        let score_bytes = checked_mul(score_elements, size_of::<f32>())?;
        if score_bytes > MAXIMUM_SCORE_BYTES {
            return Err(format!(
                "attention score scratch requires {score_bytes} bytes; limit is {MAXIMUM_SCORE_BYTES}"
            ));
        }
        let scores = self.f32_buffer(score_elements)?;
        let params = AttentionParams {
            batch: to_u32(batch, "attention batch")?,
            query_length: to_u32(query_length, "attention query length")?,
            key_length: to_u32(key_length, "attention key length")?,
            dim: to_u32(self.dim, "model dimension")?,
            heads: to_u32(self.heads, "attention heads")?,
            query_stride: to_u32(query.stride, "attention query stride")?,
            query_offset: to_u32(query.offset, "attention query offset")?,
            key_stride: to_u32(key.stride, "attention key stride")?,
            key_offset: to_u32(key.offset, "attention key offset")?,
            value_stride: to_u32(value.stride, "attention value stride")?,
            value_offset: to_u32(value.offset, "attention value offset")?,
            query_tile: to_u32(self.tuning.attention.query_tile(), "attention query tile")?,
        };
        commands.dispatch(
            &commands.runtime().pipelines.attention_scores,
            &[query.buffer, key.buffer, lengths, &scores],
            &params,
            grid(key_length, query_length, batch * self.heads),
            grid(8, 8, 1),
        )?;
        commands.dispatch_threadgroups(
            &commands.runtime().pipelines.attention_softmax,
            &[&scores],
            &params,
            grid(query_length, batch * self.heads, 1),
            grid(128, 1, 1),
        )?;
        let output = self.f32_buffer(checked_product(&[batch, query_length, self.dim])?)?;
        commands.dispatch(
            &commands.runtime().pipelines.attention_apply,
            &[&scores, value.buffer, &output],
            &params,
            grid(self.dim, query_length, batch),
            grid(32, 4, 1),
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn select_and_advance(
        &self,
        commands: &Commands<'_>,
        decoder: &Buffer,
        candidate_width: usize,
        candidate_counts: &Buffer,
        candidate_ids: &Buffer,
        previous: &Buffer,
        limits: &Buffer,
        finished: &Buffer,
        history: &Buffer,
        batch: usize,
        step: usize,
        history_step: usize,
    ) -> Result<(), String> {
        let params = SelectDecodeParams {
            batch: to_u32(batch, "decode batch")?,
            candidates: to_u32(candidate_width, "candidate width")?,
            dim: to_u32(self.dim, "model dimension")?,
            storage: self.storage.code(),
            step: to_u32(step, "decode step")?,
            history_step: to_u32(history_step, "decode history step")?,
            threads: to_u32(
                self.tuning.decode.selection_threads(),
                "decode selection threads",
            )?,
        };
        commands.dispatch_threadgroups(
            &commands.runtime().pipelines.select_decode,
            &[
                decoder,
                &self.model.decoder_embedding,
                &self.model.output_bias,
                candidate_ids,
                candidate_counts,
                previous,
                limits,
                finished,
                history,
            ],
            &params,
            grid(batch, 1, 1),
            grid(self.tuning.decode.selection_threads(), 1, 1),
        )?;
        Ok(())
    }

    fn f32_buffer(&self, elements: usize) -> Result<Buffer, String> {
        self.workspace.output_f32(&self.runtime, elements)
    }
}

fn require_f32(buffer: &Buffer, elements: usize, label: &str) -> Result<(), String> {
    let required = checked_mul(elements, size_of::<f32>())?;
    if buffer.byte_len() < required {
        return Err(format!(
            "{label} requires {required} bytes, but buffer has {}",
            buffer.byte_len()
        ));
    }
    Ok(())
}

fn require_model(
    buffer: &Buffer,
    elements: usize,
    storage: MetalStorage,
    label: &str,
) -> Result<(), String> {
    let element_bytes = match storage {
        MetalStorage::Fp32 => size_of::<f32>(),
        MetalStorage::MixedF16 => size_of::<u16>(),
    };
    let required = checked_mul(elements, element_bytes)?;
    if buffer.byte_len() < required {
        return Err(format!(
            "{label} requires {required} bytes, but buffer has {}",
            buffer.byte_len()
        ));
    }
    Ok(())
}
