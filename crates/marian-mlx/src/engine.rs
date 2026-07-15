use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    fs,
    mem::size_of,
    path::Path,
};

use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

use marian_model::{Architecture, LexicalShortlist};

use crate::metal_runtime::{Buffer, Commands, MetalParams, MetalRuntime, MetalStorage, grid};

const MAXIMUM_POSITION: usize = 4_096;
const MAXIMUM_BATCH: usize = 256;
const MAXIMUM_SCORE_BYTES: usize = 1_024 * 1_024 * 1_024;
const MATMUL_TILE: usize = 32;
const MATMUL_THREADS_Y: usize = 8;
const FLASH_ATTENTION_THREADS: usize = 32;
const FLASH_ATTENTION_MAX_HEAD_DIM: usize = 64;
const FLASH_ATTENTION_QUERY_TILE: usize = 4;
const DEFAULT_FLASH_ATTENTION_THRESHOLD: usize = 1;
const MAXIMUM_DECODE_ROWS_PER_SUBMISSION: usize = 32;
const MAXIMUM_DECODE_STEPS_PER_SUBMISSION: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttentionMode {
    Auto,
    Classic,
    Flash,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AttentionDispatch {
    mode: AttentionMode,
    threshold: usize,
}

impl AttentionDispatch {
    fn from_env() -> Result<Self, String> {
        let mode = match std::env::var("MARIAN_MLX_METAL_ATTENTION")
            .unwrap_or_else(|_| "auto".into())
            .as_str()
        {
            "auto" => AttentionMode::Auto,
            "classic" => AttentionMode::Classic,
            "flash" => AttentionMode::Flash,
            value => {
                return Err(format!(
                    "unsupported MARIAN_MLX_METAL_ATTENTION {value:?}; expected auto, classic, or flash"
                ));
            }
        };
        let threshold = std::env::var("MARIAN_MLX_METAL_FLASH_THRESHOLD")
            .ok()
            .map(|value| {
                value.parse::<usize>().map_err(|_| {
                    format!("MARIAN_MLX_METAL_FLASH_THRESHOLD {value:?} is not an integer")
                })
            })
            .transpose()?
            .unwrap_or(DEFAULT_FLASH_ATTENTION_THRESHOLD);
        if !(1..=MAXIMUM_POSITION).contains(&threshold) {
            return Err(format!(
                "MARIAN_MLX_METAL_FLASH_THRESHOLD must be between 1 and {MAXIMUM_POSITION}"
            ));
        }
        Ok(Self { mode, threshold })
    }

    fn use_flash(self, query_length: usize, key_length: usize, head_dim: usize) -> bool {
        if head_dim > FLASH_ATTENTION_MAX_HEAD_DIM {
            return false;
        }
        match self.mode {
            AttentionMode::Classic => false,
            AttentionMode::Flash => true,
            AttentionMode::Auto => {
                query_length == 1 || query_length == key_length && query_length >= self.threshold
            }
        }
    }

    fn label(self) -> String {
        match self.mode {
            AttentionMode::Classic => "classic".into(),
            AttentionMode::Flash => "flash-q4".into(),
            AttentionMode::Auto => format!("flash-q4-auto@{}", self.threshold),
        }
    }
}

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
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OutputParams {
    batch: u32,
    candidates: u32,
    dim: u32,
    storage: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AdvanceDecodeParams {
    batch: u32,
    candidates: u32,
    step: u32,
    history_step: u32,
}

// SAFETY: Every parameter is repr(C), contains only u32 fields, and mirrors
// the structure with the same name in kernels.metal.
unsafe impl MetalParams for MatMulParams {}
unsafe impl MetalParams for EmbeddingParams {}
unsafe impl MetalParams for DecoderInputParams {}
unsafe impl MetalParams for NormParams {}
unsafe impl MetalParams for AttentionParams {}
unsafe impl MetalParams for OutputParams {}
unsafe impl MetalParams for AdvanceDecodeParams {}

struct AttentionWeights {
    wq: Buffer,
    wk: Buffer,
    wv: Buffer,
    wo: Buffer,
    bq: Buffer,
    bk: Buffer,
    bv: Buffer,
    bo: Buffer,
    norm_scale: Buffer,
    norm_bias: Buffer,
}

struct FeedForwardWeights {
    w1: Buffer,
    w2: Buffer,
    b1: Buffer,
    b2: Buffer,
    norm_scale: Buffer,
    norm_bias: Buffer,
}

struct SsruWeights {
    w: Buffer,
    wf: Buffer,
    bf: Buffer,
    norm_scale: Buffer,
    norm_bias: Buffer,
}

struct EncoderLayer {
    attention: AttentionWeights,
    ffn: FeedForwardWeights,
}

struct DecoderLayer {
    ssru: SsruWeights,
    context: AttentionWeights,
    ffn: FeedForwardWeights,
}

struct ModelWeights {
    encoder_embedding: Buffer,
    decoder_embedding: Buffer,
    output_bias: Buffer,
    encoder: Vec<EncoderLayer>,
    decoder: Vec<DecoderLayer>,
}

struct CrossCache {
    key: Buffer,
    value: Buffer,
}

struct EncodedBatch {
    source_length: usize,
    lengths: Vec<usize>,
    length_buffer: Buffer,
    cross: Vec<CrossCache>,
}

struct CandidateBatch {
    width: usize,
    counts: Buffer,
    id_buffer: Buffer,
}

pub(crate) struct BatchOutput {
    pub(crate) tokens: Vec<i32>,
    pub(crate) offsets: Vec<u32>,
}

pub(crate) struct MetalEngine {
    runtime: MetalRuntime,
    healthy: Cell<bool>,
    model: ModelWeights,
    positions: Buffer,
    shortlist: LexicalShortlist,
    dummy_bias: Buffer,
    dim: usize,
    heads: usize,
    ffn_dim: usize,
    source_vocab: usize,
    max_length_factor: usize,
    scratch: RefCell<MetalBufferArena>,
    cross_scratch: RefCell<MetalBufferArena>,
    upload_scratch: RefCell<MetalUploadArena>,
    scratch_active: Cell<bool>,
    storage: MetalStorage,
    attention: AttentionDispatch,
}

#[derive(Default)]
struct MetalBufferArena {
    buffers: Vec<Buffer>,
    cursor: usize,
}

#[derive(Default)]
struct MetalUploadArena {
    buffers: Vec<Buffer>,
    cursor: usize,
}

impl MetalUploadArena {
    fn begin(&mut self) {
        self.cursor = 0;
    }

    fn take<T: crate::metal_runtime::MetalPod>(
        &mut self,
        runtime: &MetalRuntime,
        values: &[T],
    ) -> Result<Buffer, String> {
        let required = checked_mul(values.len(), size_of::<T>())?;
        let slot = self.cursor;
        self.cursor += 1;
        if slot == self.buffers.len() {
            self.buffers.push(runtime.empty::<T>(values.len())?);
        } else if self.buffers[slot].byte_len() < required {
            self.buffers[slot] = runtime.empty::<T>(values.len())?;
        }
        self.buffers[slot].write(values)?;
        Ok(self.buffers[slot].clone())
    }
}

impl MetalBufferArena {
    fn begin(&mut self) {
        self.cursor = 0;
    }

    fn take(&mut self, runtime: &MetalRuntime, elements: usize) -> Result<Buffer, String> {
        let required = checked_mul(elements, size_of::<f32>())?;
        let slot = self.cursor;
        self.cursor += 1;
        if slot == self.buffers.len() {
            self.buffers.push(runtime.empty_cached::<f32>(elements)?);
        } else if self.buffers[slot].byte_len() < required {
            self.buffers[slot] = runtime.empty_cached::<f32>(elements)?;
        }
        Ok(self.buffers[slot].clone())
    }
}

struct ScratchScope<'a> {
    active: &'a Cell<bool>,
}

impl Drop for ScratchScope<'_> {
    fn drop(&mut self) {
        self.active.set(false);
    }
}

impl MetalEngine {
    pub(crate) fn load(
        weights_path: &Path,
        shortlist_path: Option<&Path>,
        architecture: &Architecture,
    ) -> Result<Self, String> {
        let runtime = MetalRuntime::new()?;
        let attention = AttentionDispatch::from_env()?;
        let storage = runtime.storage();
        let model = ModelWeights::load(&runtime, weights_path, architecture)?;
        let positions = runtime.upload(&make_positions(architecture.model_dim))?;
        let shortlist = LexicalShortlist::load(
            shortlist_path,
            architecture.source_vocab_size,
            architecture.target_vocab_size,
        )?;
        let dummy_bias = runtime.upload(&[0.0_f32])?;
        Ok(Self {
            runtime,
            healthy: Cell::new(true),
            model,
            positions,
            shortlist,
            dummy_bias,
            dim: architecture.model_dim,
            heads: architecture.attention_heads,
            ffn_dim: architecture.ffn_dim,
            source_vocab: architecture.source_vocab_size,
            max_length_factor: architecture.max_length_factor.max(1),
            scratch: RefCell::new(MetalBufferArena::default()),
            cross_scratch: RefCell::new(MetalBufferArena::default()),
            upload_scratch: RefCell::new(MetalUploadArena::default()),
            scratch_active: Cell::new(false),
            storage,
            attention,
        })
    }

    pub(crate) fn device_name(&self) -> &str {
        self.runtime.device_name()
    }

    pub(crate) fn precision(&self) -> &'static str {
        self.storage.label()
    }

    pub(crate) fn attention_label(&self) -> String {
        self.attention.label()
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.healthy.get()
    }

    pub(crate) fn warmup(&self) -> Result<(), String> {
        self.translate(&[2, 0], &[0, 2], &[4]).map(|_| ())
    }

    pub(crate) fn translate(
        &self,
        tokens: &[i32],
        offsets: &[u32],
        max_output_tokens: &[usize],
    ) -> Result<BatchOutput, String> {
        self.validate_batch(tokens, offsets, max_output_tokens)?;
        self.upload_scratch.borrow_mut().begin();
        let batch = offsets.len() - 1;
        let encoded = self.encode(tokens, offsets)?;
        let candidates = self.prepare_candidates(tokens, offsets)?;
        let states = (0..self.model.decoder.len())
            .map(|_| self.upload_buffer(&vec![0.0_f32; checked_mul(batch, self.dim)?]))
            .collect::<Result<Vec<_>, String>>()?;
        let previous_buffer = self.upload_buffer(&vec![-1_i32; batch])?;
        let mut generated = vec![Vec::<i32>::new(); batch];
        let limits = encoded
            .lengths
            .iter()
            .zip(max_output_tokens)
            .map(|(&length, &requested)| {
                requested
                    .min(length.saturating_mul(self.max_length_factor))
                    .min(MAXIMUM_POSITION)
            })
            .collect::<Vec<_>>();
        let generation_limit = limits.iter().copied().max().unwrap_or(0);
        let gpu_limits = limits
            .iter()
            .map(|&limit| to_u32(limit, "output token limit"))
            .collect::<Result<Vec<_>, _>>()?;
        let limit_buffer = self.upload_buffer(&gpu_limits)?;
        let finished_buffer = self.upload_buffer(&vec![0_u32; batch])?;
        let mut step = 0;

        while step < generation_limit {
            let steps_in_submission = (MAXIMUM_DECODE_ROWS_PER_SUBMISSION / batch)
                .clamp(1, MAXIMUM_DECODE_STEPS_PER_SUBMISSION)
                .min(generation_limit - step);
            let history_buffer =
                self.upload_buffer(&vec![-1_i32; checked_mul(steps_in_submission, batch)?])?;
            let _scratch = self.begin_scratch()?;
            let commands = self.commands()?;
            for history_step in 0..steps_in_submission {
                let absolute_step = step + history_step;
                let mut decoder =
                    self.decoder_input(&commands, &previous_buffer, batch, absolute_step)?;

                for (index, layer) in self.model.decoder.iter().enumerate() {
                    let update = self.matmul(
                        &commands,
                        &decoder,
                        &layer.ssru.w,
                        None,
                        batch,
                        self.dim,
                        self.dim,
                    )?;
                    let forget = self.matmul(
                        &commands,
                        &decoder,
                        &layer.ssru.wf,
                        Some(&layer.ssru.bf),
                        batch,
                        self.dim,
                        self.dim,
                    )?;
                    decoder = self.ssru_norm(
                        &commands,
                        &update,
                        &forget,
                        &states[index],
                        &decoder,
                        &layer.ssru.norm_scale,
                        &layer.ssru.norm_bias,
                        batch,
                    )?;

                    let query = self.matmul(
                        &commands,
                        &decoder,
                        &layer.context.wq,
                        Some(&layer.context.bq),
                        batch,
                        self.dim,
                        self.dim,
                    )?;
                    let attended = self.attend(
                        &commands,
                        &query,
                        &encoded.cross[index].key,
                        &encoded.cross[index].value,
                        &encoded.length_buffer,
                        batch,
                        1,
                        encoded.source_length,
                    )?;
                    let context = self.matmul(
                        &commands,
                        &attended,
                        &layer.context.wo,
                        Some(&layer.context.bo),
                        batch,
                        self.dim,
                        self.dim,
                    )?;
                    decoder = self.residual_norm(
                        &commands,
                        &context,
                        &decoder,
                        &layer.context.norm_scale,
                        &layer.context.norm_bias,
                        batch,
                    )?;
                    decoder = self.feed_forward(&commands, &decoder, &layer.ffn, batch)?;
                }

                let selected = self.select_tokens(&commands, &decoder, &candidates, batch)?;
                let params = AdvanceDecodeParams {
                    batch: to_u32(batch, "decode batch")?,
                    candidates: to_u32(candidates.width, "candidate width")?,
                    step: to_u32(absolute_step, "decode step")?,
                    history_step: to_u32(history_step, "decode history step")?,
                };
                commands.dispatch(
                    &commands.runtime().pipelines.advance_decode,
                    &[
                        &selected,
                        &candidates.id_buffer,
                        &previous_buffer,
                        &limit_buffer,
                        &finished_buffer,
                        &history_buffer,
                    ],
                    &params,
                    grid(batch, 1, 1),
                    grid(64, 1, 1),
                );
            }
            self.finish_commands(commands)?;
            let history = history_buffer.read::<i32>(checked_mul(steps_in_submission, batch)?)?;
            for history_step in 0..steps_in_submission {
                for row in 0..batch {
                    let token = history[history_step * batch + row];
                    if token >= 0 {
                        generated[row].push(token);
                    }
                }
            }
            step += steps_in_submission;
            if finished_buffer
                .read::<u32>(batch)?
                .iter()
                .all(|&value| value != 0)
            {
                break;
            }
        }

        let mut output = BatchOutput {
            tokens: Vec::new(),
            offsets: Vec::with_capacity(batch + 1),
        };
        output.offsets.push(0);
        for sentence in generated {
            output.tokens.extend(sentence);
            output.offsets.push(
                u32::try_from(output.tokens.len())
                    .map_err(|_| "generated token count exceeds u32".to_string())?,
            );
        }
        Ok(output)
    }

    fn validate_batch(
        &self,
        tokens: &[i32],
        offsets: &[u32],
        max_output_tokens: &[usize],
    ) -> Result<(), String> {
        if offsets.len() < 2
            || offsets[0] != 0
            || offsets.last().copied() != Some(tokens.len() as u32)
        {
            return Err("invalid packed batch offsets".into());
        }
        let batch = offsets.len() - 1;
        if batch > MAXIMUM_BATCH {
            return Err(format!(
                "batch contains {batch} sentences; maximum is {MAXIMUM_BATCH}"
            ));
        }
        if max_output_tokens.len() != batch {
            return Err("max_output_tokens does not match packed batch".into());
        }
        for pair in offsets.windows(2) {
            if pair[1] <= pair[0] {
                return Err("each input must contain at least one token".into());
            }
        }
        for &token in tokens {
            if token < 0 || token as usize >= self.source_vocab {
                return Err(format!(
                    "source token {token} is outside vocabulary {}",
                    self.source_vocab
                ));
            }
        }
        Ok(())
    }

    fn encode(&self, tokens: &[i32], offsets: &[u32]) -> Result<EncodedBatch, String> {
        let batch = offsets.len() - 1;
        let lengths = offsets
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as usize)
            .collect::<Vec<_>>();
        let source_length = lengths.iter().copied().max().unwrap_or(0);
        if source_length > MAXIMUM_POSITION {
            return Err(format!(
                "source has {source_length} tokens; maximum is {MAXIMUM_POSITION}"
            ));
        }
        let mut padded = vec![0_i32; checked_mul(batch, source_length)?];
        for row in 0..batch {
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            padded[row * source_length..row * source_length + lengths[row]]
                .copy_from_slice(&tokens[start..end]);
        }
        let gpu_lengths = lengths
            .iter()
            .map(|&length| to_u32(length, "source length"))
            .collect::<Result<Vec<_>, _>>()?;
        let token_buffer = self.upload_buffer(&padded)?;
        let length_buffer = self.upload_buffer(&gpu_lengths)?;

        // Flash attention does not retain the classic path's quadratic score
        // buffers, so the complete encoder and cross-cache projection can be
        // submitted in one command buffer. This removes seven CPU/GPU waits
        // from every request batch while preserving the layer order recorded
        // in the compute encoder.
        if self
            .attention
            .use_flash(source_length, source_length, self.dim / self.heads)
        {
            let _scratch = self.begin_scratch()?;
            let commands = self.commands()?;
            let mut encoder = self.embedding(&commands, &token_buffer, batch, source_length)?;
            for layer in &self.model.encoder {
                encoder = self.encode_layer(
                    &commands,
                    &encoder,
                    layer,
                    &length_buffer,
                    batch,
                    source_length,
                )?;
            }
            let cross = self.build_cross_cache(&commands, &encoder, batch, source_length)?;
            self.finish_commands(commands)?;
            return Ok(EncodedBatch {
                source_length,
                lengths,
                length_buffer,
                cross,
            });
        }

        let commands = self.commands()?;
        let mut encoder = self.embedding(&commands, &token_buffer, batch, source_length)?;
        self.finish_commands(commands)?;

        for layer in &self.model.encoder {
            // Finish each encoder layer before dispatching the next one. A
            // retained command buffer keeps every temporary attention score
            // allocation alive; one command buffer for all six layers would
            // otherwise multiply the O(sequence^2) scratch requirement.
            let commands = self.commands()?;
            encoder = self.encode_layer(
                &commands,
                &encoder,
                layer,
                &length_buffer,
                batch,
                source_length,
            )?;
            self.finish_commands(commands)?;
        }

        let commands = self.commands()?;
        let cross = self.build_cross_cache(&commands, &encoder, batch, source_length)?;
        self.finish_commands(commands)?;
        Ok(EncodedBatch {
            source_length,
            lengths,
            length_buffer,
            cross,
        })
    }

    fn encode_layer(
        &self,
        commands: &Commands<'_>,
        encoder: &Buffer,
        layer: &EncoderLayer,
        lengths: &Buffer,
        batch: usize,
        source_length: usize,
    ) -> Result<Buffer, String> {
        let rows = checked_mul(batch, source_length)?;
        let query = self.matmul(
            commands,
            encoder,
            &layer.attention.wq,
            Some(&layer.attention.bq),
            rows,
            self.dim,
            self.dim,
        )?;
        let key = self.matmul(
            commands,
            encoder,
            &layer.attention.wk,
            Some(&layer.attention.bk),
            rows,
            self.dim,
            self.dim,
        )?;
        let value = self.matmul(
            commands,
            encoder,
            &layer.attention.wv,
            Some(&layer.attention.bv),
            rows,
            self.dim,
            self.dim,
        )?;
        let attended = self.attend(
            commands,
            &query,
            &key,
            &value,
            lengths,
            batch,
            source_length,
            source_length,
        )?;
        let projected = self.matmul(
            commands,
            &attended,
            &layer.attention.wo,
            Some(&layer.attention.bo),
            rows,
            self.dim,
            self.dim,
        )?;
        let normalized = self.residual_norm(
            commands,
            &projected,
            encoder,
            &layer.attention.norm_scale,
            &layer.attention.norm_bias,
            rows,
        )?;
        self.feed_forward(commands, &normalized, &layer.ffn, rows)
    }

    fn build_cross_cache(
        &self,
        commands: &Commands<'_>,
        encoder: &Buffer,
        batch: usize,
        source_length: usize,
    ) -> Result<Vec<CrossCache>, String> {
        self.cross_scratch.borrow_mut().begin();
        let rows = checked_mul(batch, source_length)?;
        let mut cross = Vec::with_capacity(self.model.decoder.len());
        for layer in &self.model.decoder {
            let key = self.matmul_persistent(
                commands,
                encoder,
                &layer.context.wk,
                Some(&layer.context.bk),
                rows,
                self.dim,
                self.dim,
            )?;
            let value = self.matmul_persistent(
                commands,
                encoder,
                &layer.context.wv,
                Some(&layer.context.bv),
                rows,
                self.dim,
                self.dim,
            )?;
            cross.push(CrossCache { key, value });
        }
        Ok(cross)
    }

    fn finish_commands(&self, commands: Commands<'_>) -> Result<(), String> {
        if let Err(error) = commands.finish() {
            // A command-buffer failure means the queue can no longer be
            // trusted for subsequent requests. Let the scheduler reject new
            // work instead of repeatedly dispatching into a failed device.
            self.healthy.set(false);
            return Err(error);
        }
        Ok(())
    }

    fn commands(&self) -> Result<Commands<'_>, String> {
        match self.runtime.commands() {
            Ok(commands) => Ok(commands),
            Err(error) => {
                self.healthy.set(false);
                Err(error)
            }
        }
    }

    fn begin_scratch(&self) -> Result<ScratchScope<'_>, String> {
        if self.scratch_active.replace(true) {
            return Err("nested Metal scratch scopes are not supported".into());
        }
        self.scratch.borrow_mut().begin();
        Ok(ScratchScope {
            active: &self.scratch_active,
        })
    }

    fn prepare_candidates(
        &self,
        tokens: &[i32],
        offsets: &[u32],
    ) -> Result<CandidateBatch, String> {
        let rows = offsets
            .windows(2)
            .map(|pair| {
                self.shortlist
                    .candidates(&tokens[pair[0] as usize..pair[1] as usize])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let width = rows.iter().map(Vec::len).max().unwrap_or(0);
        if width == 0 {
            return Err("lexical shortlist produced an empty batch".into());
        }
        let counts = rows
            .iter()
            .map(|row| to_u32(row.len(), "candidate count"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut ids = vec![0_u32; checked_mul(rows.len(), width)?];
        for (row_index, row) in rows.iter().enumerate() {
            ids[row_index * width..row_index * width + row.len()].copy_from_slice(row);
        }
        Ok(CandidateBatch {
            id_buffer: self.upload_buffer(&ids)?,
            counts: self.upload_buffer(&counts)?,
            width,
        })
    }

    fn embedding(
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
        );
        Ok(output)
    }

    fn decoder_input(
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
        );
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul(
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
    fn matmul_persistent(
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
    fn matmul_with_activation(
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
            self.cross_scratch
                .borrow_mut()
                .take(&self.runtime, elements)?
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
        if self.storage == MetalStorage::Fp32 {
            commands.mps_matmul(lhs, rhs, &output, rows, cols, inner)?;
            commands.dispatch(
                &commands.runtime().pipelines.matmul_bias_activation,
                &[&output, bias.unwrap_or(&self.dummy_bias)],
                &params,
                grid(cols, rows, 1),
                grid(32, 8, 1),
            );
        } else {
            commands.dispatch_threadgroups(
                &commands.runtime().pipelines.matmul_microtile,
                &[lhs, rhs, bias.unwrap_or(&self.dummy_bias), &output],
                &params,
                grid(cols.div_ceil(MATMUL_TILE), rows.div_ceil(MATMUL_TILE), 1),
                grid(MATMUL_TILE, MATMUL_THREADS_Y, 1),
            );
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn residual_norm(
        &self,
        commands: &Commands<'_>,
        input: &Buffer,
        residual: &Buffer,
        scale: &Buffer,
        bias: &Buffer,
        rows: usize,
    ) -> Result<Buffer, String> {
        let elements = checked_mul(rows, self.dim)?;
        require_f32(input, elements, "layer norm input")?;
        require_f32(residual, elements, "layer norm residual")?;
        require_model(scale, self.dim, self.storage, "layer norm scale")?;
        require_model(bias, self.dim, self.storage, "layer norm bias")?;
        let output = self.f32_buffer(elements)?;
        let params = NormParams {
            rows: to_u32(rows, "layer norm rows")?,
            dim: to_u32(self.dim, "model dimension")?,
            storage: self.storage.code(),
        };
        commands.dispatch_threadgroups(
            &commands.runtime().pipelines.residual_norm,
            &[input, residual, scale, bias, &output],
            &params,
            grid(rows, 1, 1),
            grid(128, 1, 1),
        );
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn ssru_norm(
        &self,
        commands: &Commands<'_>,
        update: &Buffer,
        forget: &Buffer,
        state: &Buffer,
        residual: &Buffer,
        scale: &Buffer,
        bias: &Buffer,
        rows: usize,
    ) -> Result<Buffer, String> {
        let elements = checked_mul(rows, self.dim)?;
        for (buffer, label) in [
            (update, "SSRU update"),
            (forget, "SSRU forget"),
            (state, "SSRU state"),
            (residual, "SSRU residual"),
        ] {
            require_f32(buffer, elements, label)?;
        }
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
            &[update, forget, state, residual, scale, bias, &output],
            &params,
            grid(rows, 1, 1),
            grid(128, 1, 1),
        );
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        commands: &Commands<'_>,
        query: &Buffer,
        key: &Buffer,
        value: &Buffer,
        lengths: &Buffer,
        batch: usize,
        query_length: usize,
        key_length: usize,
    ) -> Result<Buffer, String> {
        let head_dim = self.dim / self.heads;
        if self.attention.use_flash(query_length, key_length, head_dim) {
            let output = self.f32_buffer(checked_product(&[batch, query_length, self.dim])?)?;
            let params = AttentionParams {
                batch: to_u32(batch, "attention batch")?,
                query_length: to_u32(query_length, "attention query length")?,
                key_length: to_u32(key_length, "attention key length")?,
                dim: to_u32(self.dim, "model dimension")?,
                heads: to_u32(self.heads, "attention heads")?,
            };
            commands.dispatch_threadgroups(
                &commands.runtime().pipelines.attention_flash,
                &[query, key, lengths, value, &output],
                &params,
                grid(
                    query_length.div_ceil(FLASH_ATTENTION_QUERY_TILE),
                    batch * self.heads,
                    1,
                ),
                grid(FLASH_ATTENTION_THREADS, 1, 1),
            );
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
        };
        commands.dispatch(
            &commands.runtime().pipelines.attention_scores,
            &[query, key, lengths, &scores],
            &params,
            grid(key_length, query_length, batch * self.heads),
            grid(8, 8, 1),
        );
        commands.dispatch_threadgroups(
            &commands.runtime().pipelines.attention_softmax,
            &[&scores],
            &params,
            grid(query_length, batch * self.heads, 1),
            grid(128, 1, 1),
        );
        let output = self.f32_buffer(checked_product(&[batch, query_length, self.dim])?)?;
        commands.dispatch(
            &commands.runtime().pipelines.attention_apply,
            &[&scores, value, &output],
            &params,
            grid(self.dim, query_length, batch),
            grid(32, 4, 1),
        );
        Ok(output)
    }

    fn feed_forward(
        &self,
        commands: &Commands<'_>,
        input: &Buffer,
        weights: &FeedForwardWeights,
        rows: usize,
    ) -> Result<Buffer, String> {
        let hidden = self.matmul_with_activation(
            commands,
            input,
            &weights.w1,
            Some(&weights.b1),
            rows,
            self.ffn_dim,
            self.dim,
            true,
            false,
        )?;
        let output = self.matmul(
            commands,
            &hidden,
            &weights.w2,
            Some(&weights.b2),
            rows,
            self.dim,
            self.ffn_dim,
        )?;
        self.residual_norm(
            commands,
            &output,
            input,
            &weights.norm_scale,
            &weights.norm_bias,
            rows,
        )
    }

    fn select_tokens(
        &self,
        commands: &Commands<'_>,
        decoder: &Buffer,
        candidates: &CandidateBatch,
        batch: usize,
    ) -> Result<Buffer, String> {
        let logits = self.f32_buffer(checked_mul(batch, candidates.width)?)?;
        let params = OutputParams {
            batch: to_u32(batch, "output batch")?,
            candidates: to_u32(candidates.width, "candidate width")?,
            dim: to_u32(self.dim, "model dimension")?,
            storage: self.storage.code(),
        };
        commands.dispatch(
            &commands.runtime().pipelines.output_logits,
            &[
                decoder,
                &self.model.decoder_embedding,
                &self.model.output_bias,
                &candidates.id_buffer,
                &candidates.counts,
                &logits,
            ],
            &params,
            grid(candidates.width, batch, 1),
            grid(16, 8, 1),
        );
        let selected = self.runtime.empty::<u32>(batch)?;
        commands.dispatch_threadgroups(
            &commands.runtime().pipelines.argmax,
            &[&logits, &candidates.counts, &selected],
            &params,
            grid(batch, 1, 1),
            grid(128, 1, 1),
        );
        Ok(selected)
    }

    fn f32_buffer(&self, elements: usize) -> Result<Buffer, String> {
        if self.scratch_active.get() {
            self.scratch.borrow_mut().take(&self.runtime, elements)
        } else {
            self.runtime.empty::<f32>(elements)
        }
    }

    fn upload_buffer<T: crate::metal_runtime::MetalPod>(
        &self,
        values: &[T],
    ) -> Result<Buffer, String> {
        self.upload_scratch.borrow_mut().take(&self.runtime, values)
    }
}

impl ModelWeights {
    fn load(
        runtime: &MetalRuntime,
        path: &Path,
        architecture: &Architecture,
    ) -> Result<Self, String> {
        const MAXIMUM_WEIGHT_FILE_BYTES: u64 = 1_024 * 1_024 * 1_024;
        let metadata = fs::metadata(path)
            .map_err(|error| format!("failed to inspect weights {}: {error}", path.display()))?;
        if metadata.len() > MAXIMUM_WEIGHT_FILE_BYTES {
            return Err(format!(
                "weights {} contain {} bytes; maximum is {MAXIMUM_WEIGHT_FILE_BYTES}",
                path.display(),
                metadata.len()
            ));
        }
        let file = fs::File::open(path)
            .map_err(|error| format!("failed to open weights {}: {error}", path.display()))?;
        // SAFETY: Read-only mapping; Metal uploads every tensor before this
        // function returns, so no Buffer retains a pointer into the mapping.
        let bytes = unsafe { MmapOptions::new().map(&file) }
            .map_err(|error| format!("failed to map weights {}: {error}", path.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)
            .map_err(|error| format!("invalid safetensors {}: {error}", path.display()))?;
        let mut tensors = tensors.tensors().into_iter().collect::<HashMap<_, _>>();
        let d = architecture.model_dim;
        let f = architecture.ffn_dim;
        let source_vocab = architecture.source_vocab_size;
        let target_vocab = architecture.target_vocab_size;

        let encoder_embedding =
            take_tensor(runtime, &mut tensors, "encoder_Wemb", &[source_vocab, d])?;
        let decoder_embedding =
            take_tensor(runtime, &mut tensors, "decoder_Wemb", &[target_vocab, d])?;
        let output_bias = take_tensor(
            runtime,
            &mut tensors,
            "decoder_ff_logit_out_b",
            &[1, target_vocab],
        )?;
        let mut encoder = Vec::with_capacity(architecture.encoder_layers);
        for layer in 1..=architecture.encoder_layers {
            encoder.push(EncoderLayer {
                attention: load_attention(
                    runtime,
                    &mut tensors,
                    &format!("encoder_l{layer}_self"),
                    d,
                )?,
                ffn: load_ffn(
                    runtime,
                    &mut tensors,
                    &format!("encoder_l{layer}_ffn"),
                    d,
                    f,
                )?,
            });
        }
        let mut decoder = Vec::with_capacity(architecture.decoder_layers);
        for layer in 1..=architecture.decoder_layers {
            decoder.push(DecoderLayer {
                ssru: load_ssru(runtime, &mut tensors, &format!("decoder_l{layer}_rnn"), d)?,
                context: load_attention(
                    runtime,
                    &mut tensors,
                    &format!("decoder_l{layer}_context"),
                    d,
                )?,
                ffn: load_ffn(
                    runtime,
                    &mut tensors,
                    &format!("decoder_l{layer}_ffn"),
                    d,
                    f,
                )?,
            });
        }
        if !tensors.is_empty() {
            let mut names = tensors.keys().cloned().collect::<Vec<_>>();
            names.sort();
            return Err(format!(
                "safetensors contains {} unexpected tensors: {}",
                names.len(),
                names.into_iter().take(8).collect::<Vec<_>>().join(", ")
            ));
        }
        Ok(Self {
            encoder_embedding,
            decoder_embedding,
            output_bias,
            encoder,
            decoder,
        })
    }
}

fn load_attention(
    runtime: &MetalRuntime,
    tensors: &mut HashMap<String, TensorView<'_>>,
    prefix: &str,
    dim: usize,
) -> Result<AttentionWeights, String> {
    Ok(AttentionWeights {
        wq: take_tensor(runtime, tensors, &format!("{prefix}_Wq"), &[dim, dim])?,
        wk: take_tensor(runtime, tensors, &format!("{prefix}_Wk"), &[dim, dim])?,
        wv: take_tensor(runtime, tensors, &format!("{prefix}_Wv"), &[dim, dim])?,
        wo: take_tensor(runtime, tensors, &format!("{prefix}_Wo"), &[dim, dim])?,
        bq: take_tensor(runtime, tensors, &format!("{prefix}_bq"), &[1, dim])?,
        bk: take_tensor(runtime, tensors, &format!("{prefix}_bk"), &[1, dim])?,
        bv: take_tensor(runtime, tensors, &format!("{prefix}_bv"), &[1, dim])?,
        bo: take_tensor(runtime, tensors, &format!("{prefix}_bo"), &[1, dim])?,
        norm_scale: take_tensor(
            runtime,
            tensors,
            &format!("{prefix}_Wo_ln_scale"),
            &[1, dim],
        )?,
        norm_bias: take_tensor(runtime, tensors, &format!("{prefix}_Wo_ln_bias"), &[1, dim])?,
    })
}

fn load_ffn(
    runtime: &MetalRuntime,
    tensors: &mut HashMap<String, TensorView<'_>>,
    prefix: &str,
    dim: usize,
    ffn_dim: usize,
) -> Result<FeedForwardWeights, String> {
    Ok(FeedForwardWeights {
        w1: take_tensor(runtime, tensors, &format!("{prefix}_W1"), &[dim, ffn_dim])?,
        w2: take_tensor(runtime, tensors, &format!("{prefix}_W2"), &[ffn_dim, dim])?,
        b1: take_tensor(runtime, tensors, &format!("{prefix}_b1"), &[1, ffn_dim])?,
        b2: take_tensor(runtime, tensors, &format!("{prefix}_b2"), &[1, dim])?,
        norm_scale: take_tensor(
            runtime,
            tensors,
            &format!("{prefix}_ffn_ln_scale"),
            &[1, dim],
        )?,
        norm_bias: take_tensor(
            runtime,
            tensors,
            &format!("{prefix}_ffn_ln_bias"),
            &[1, dim],
        )?,
    })
}

fn load_ssru(
    runtime: &MetalRuntime,
    tensors: &mut HashMap<String, TensorView<'_>>,
    prefix: &str,
    dim: usize,
) -> Result<SsruWeights, String> {
    Ok(SsruWeights {
        w: take_tensor(runtime, tensors, &format!("{prefix}_W"), &[dim, dim])?,
        wf: take_tensor(runtime, tensors, &format!("{prefix}_Wf"), &[dim, dim])?,
        bf: take_tensor(runtime, tensors, &format!("{prefix}_bf"), &[1, dim])?,
        norm_scale: take_tensor(
            runtime,
            tensors,
            &format!("{prefix}_ffn_ln_scale"),
            &[1, dim],
        )?,
        norm_bias: take_tensor(
            runtime,
            tensors,
            &format!("{prefix}_ffn_ln_bias"),
            &[1, dim],
        )?,
    })
}

fn take_tensor(
    runtime: &MetalRuntime,
    tensors: &mut HashMap<String, TensorView<'_>>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Buffer, String> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| format!("missing model tensor {name}"))?;
    if tensor.dtype() != Dtype::F32 {
        return Err(format!(
            "model tensor {name} has dtype {:?}; direct Metal v1 requires F32",
            tensor.dtype()
        ));
    }
    if tensor.shape() != expected_shape {
        return Err(format!(
            "model tensor {name} has shape {:?}; expected {expected_shape:?}",
            tensor.shape()
        ));
    }
    runtime.upload_model_f32(tensor.data())
}

fn make_positions(dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut values = vec![0.0_f32; MAXIMUM_POSITION * dim];
    for position in 0..MAXIMUM_POSITION {
        for index in 0..half {
            let frequency = (-(index as f32) * 10_000.0_f32.ln() / (half - 1) as f32).exp();
            values[position * dim + index] = (position as f32 * frequency).sin();
            values[position * dim + half + index] = (position as f32 * frequency).cos();
        }
    }
    values
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

fn checked_product(values: &[usize]) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1_usize, |product, &value| checked_mul(product, value))
}

fn checked_mul(lhs: usize, rhs: usize) -> Result<usize, String> {
    lhs.checked_mul(rhs)
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Metal tensor shape {lhs} x {rhs} is zero or overflows"))
}

fn to_u32(value: usize, label: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{label} {value} exceeds u32"))
}

#[cfg(test)]
mod tests {
    use super::{AttentionDispatch, AttentionMode, make_positions};

    #[test]
    fn attention_dispatch_respects_mode_shape_threshold_and_head_limit() {
        let classic = AttentionDispatch {
            mode: AttentionMode::Classic,
            threshold: 1,
        };
        assert!(!classic.use_flash(128, 128, 48));

        let flash = AttentionDispatch {
            mode: AttentionMode::Flash,
            threshold: 4_096,
        };
        assert!(flash.use_flash(7, 19, 48));
        assert!(!flash.use_flash(7, 19, 65));

        let auto = AttentionDispatch {
            mode: AttentionMode::Auto,
            threshold: 128,
        };
        assert!(auto.use_flash(1, 512, 48));
        assert!(!auto.use_flash(127, 127, 48));
        assert!(auto.use_flash(128, 128, 48));
        assert!(!auto.use_flash(128, 256, 48));
    }

    #[test]
    fn sinusoidal_positions_are_grouped_sin_then_cos() {
        let positions = make_positions(384);
        assert_eq!(positions[0], 0.0);
        assert_eq!(positions[191], 0.0);
        assert_eq!(positions[192], 1.0);
        assert_eq!(positions[383], 1.0);
        assert!((positions[384] - 1.0_f32.sin()).abs() < 1.0e-7);
        assert!((positions[384 + 192] - 1.0_f32.cos()).abs() < 1.0e-7);
    }
}
