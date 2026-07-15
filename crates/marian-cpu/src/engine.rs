use std::{collections::HashMap, fs, path::Path};

use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

use marian_model::{Architecture, LexicalShortlist, MAXIMUM_POSITION, sinusoidal_positions};

use crate::limits::{
    MAXIMUM_ENGINE_BATCH, MAXIMUM_GENERATION_STEPS, MAXIMUM_PADDED_ATTENTION_CELLS,
    MAXIMUM_SOURCE_TOKENS as MAXIMUM_SOURCE_LENGTH,
};
use crate::tensor::{
    Matrix, attention, matmul, relu_in_place, residual_layer_norm, select_token,
    ssru_update_layer_norm,
};

const MAXIMUM_WEIGHT_FILE_BYTES: u64 = 256 * 1_024 * 1_024;
const EMBEDDING_SCALE: f32 = 19.595_919;

struct AttentionWeights {
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
    bq: Matrix,
    bk: Matrix,
    bv: Matrix,
    bo: Matrix,
    norm_scale: Matrix,
    norm_bias: Matrix,
}

struct FeedForwardWeights {
    w1: Matrix,
    w2: Matrix,
    b1: Matrix,
    b2: Matrix,
    norm_scale: Matrix,
    norm_bias: Matrix,
}

struct SsruWeights {
    w: Matrix,
    wf: Matrix,
    bf: Matrix,
    norm_scale: Matrix,
    norm_bias: Matrix,
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
    encoder_embedding: Matrix,
    decoder_embedding: Matrix,
    output_bias: Matrix,
    encoder: Vec<EncoderLayer>,
    decoder: Vec<DecoderLayer>,
}

struct CrossCache {
    key: Vec<f32>,
    value: Vec<f32>,
}

struct EncodedBatch {
    source_length: usize,
    lengths: Vec<usize>,
    cross: Vec<CrossCache>,
}

/// Packed token IDs produced by one inference batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchOutput {
    pub tokens: Vec<i32>,
    pub offsets: Vec<u32>,
}

/// Pure Rust FP32 executor for the fixed 384d Transformer-SSRU graph.
///
/// Tokenization intentionally lives outside this type. Callers can use this
/// API for differential tests without bringing any native tokenizer into the
/// production dependency graph.
pub struct CpuEngine {
    model: ModelWeights,
    positions: Vec<f32>,
    shortlist: LexicalShortlist,
    dim: usize,
    heads: usize,
    ffn_dim: usize,
    source_vocab: usize,
    max_length_factor: usize,
}

impl CpuEngine {
    pub fn load(
        weights_path: impl AsRef<Path>,
        shortlist_path: Option<&Path>,
        architecture: &Architecture,
    ) -> Result<Self, String> {
        architecture.validate_supported()?;
        let model = ModelWeights::load(weights_path.as_ref(), architecture)?;
        let shortlist = LexicalShortlist::load(
            shortlist_path,
            architecture.source_vocab_size,
            architecture.target_vocab_size,
        )?;
        Ok(Self {
            model,
            positions: sinusoidal_positions(architecture.model_dim)?,
            shortlist,
            dim: architecture.model_dim,
            heads: architecture.attention_heads,
            ffn_dim: architecture.ffn_dim,
            source_vocab: architecture.source_vocab_size,
            max_length_factor: architecture.max_length_factor.max(1),
        })
    }

    pub fn warmup(&self) -> Result<(), String> {
        self.translate_token_ids(&[2, 0], &[0, 2], &[4]).map(|_| ())
    }

    pub fn translate_token_ids(
        &self,
        tokens: &[i32],
        offsets: &[u32],
        max_output_tokens: &[usize],
    ) -> Result<BatchOutput, String> {
        self.validate_batch(tokens, offsets, max_output_tokens)?;
        let batch = offsets.len() - 1;
        let encoded = self.encode(tokens, offsets)?;
        let candidates = self.prepare_candidates(tokens, offsets)?;
        let mut states = (0..self.model.decoder.len())
            .map(|_| vec![0.0_f32; batch * self.dim])
            .collect::<Vec<_>>();
        let mut previous = vec![-1_i32; batch];
        let mut finished = vec![false; batch];
        let mut generated = vec![Vec::<i32>::new(); batch];
        let limits = encoded
            .lengths
            .iter()
            .zip(max_output_tokens)
            .map(|(&length, &requested)| {
                requested
                    .min(length.saturating_mul(self.max_length_factor))
                    .min(MAXIMUM_GENERATION_STEPS)
                    .min(MAXIMUM_POSITION)
            })
            .collect::<Vec<_>>();
        let generation_limit = limits.iter().copied().max().unwrap_or(0);

        for step in 0..generation_limit {
            for row in 0..batch {
                if !finished[row] && step >= limits[row] {
                    finished[row] = true;
                }
            }
            if finished.iter().all(|value| *value) {
                break;
            }

            let mut decoder = self.decoder_input(&previous, batch, step)?;
            for (index, layer) in self.model.decoder.iter().enumerate() {
                let update = matmul(&decoder, &layer.ssru.w, batch, self.dim, None)?;
                let forget = matmul(
                    &decoder,
                    &layer.ssru.wf,
                    batch,
                    self.dim,
                    Some(&layer.ssru.bf),
                )?;
                decoder = ssru_update_layer_norm(
                    &update,
                    &forget,
                    &mut states[index],
                    &decoder,
                    &layer.ssru.norm_scale,
                    &layer.ssru.norm_bias,
                    batch,
                    self.dim,
                )?;

                let query = matmul(
                    &decoder,
                    &layer.context.wq,
                    batch,
                    self.dim,
                    Some(&layer.context.bq),
                )?;
                let attended = attention(
                    &query,
                    &encoded.cross[index].key,
                    &encoded.cross[index].value,
                    &encoded.lengths,
                    batch,
                    1,
                    encoded.source_length,
                    self.dim,
                    self.heads,
                )?;
                let context = matmul(
                    &attended,
                    &layer.context.wo,
                    batch,
                    self.dim,
                    Some(&layer.context.bo),
                )?;
                decoder = residual_layer_norm(
                    &context,
                    &decoder,
                    &layer.context.norm_scale,
                    &layer.context.norm_bias,
                    batch,
                    self.dim,
                )?;
                decoder = self.feed_forward(&decoder, &layer.ffn, batch)?;
            }

            for row in 0..batch {
                if finished[row] {
                    continue;
                }
                let state = &decoder[row * self.dim..(row + 1) * self.dim];
                let token = select_token(
                    state,
                    &self.model.decoder_embedding,
                    &self.model.output_bias,
                    &candidates[row],
                )? as i32;
                previous[row] = token;
                generated[row].push(token);
                if token == 0 {
                    finished[row] = true;
                }
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
            || offsets.last().copied() != u32::try_from(tokens.len()).ok()
        {
            return Err("invalid packed batch offsets".into());
        }
        let batch = offsets.len() - 1;
        if batch > MAXIMUM_ENGINE_BATCH {
            return Err(format!(
                "batch contains {batch} sentences; maximum is {MAXIMUM_ENGINE_BATCH}"
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
        let source_length = offsets
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as usize)
            .max()
            .unwrap_or(0);
        validate_work_shape(batch, source_length)?;
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
        let mut padded = vec![0_i32; batch * source_length];
        for row in 0..batch {
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            padded[row * source_length..row * source_length + lengths[row]]
                .copy_from_slice(&tokens[start..end]);
        }
        let mut encoder = self.embedding(&padded, batch, source_length)?;
        for layer in &self.model.encoder {
            let rows = batch * source_length;
            let query = matmul(
                &encoder,
                &layer.attention.wq,
                rows,
                self.dim,
                Some(&layer.attention.bq),
            )?;
            let key = matmul(
                &encoder,
                &layer.attention.wk,
                rows,
                self.dim,
                Some(&layer.attention.bk),
            )?;
            let value = matmul(
                &encoder,
                &layer.attention.wv,
                rows,
                self.dim,
                Some(&layer.attention.bv),
            )?;
            let attended = attention(
                &query,
                &key,
                &value,
                &lengths,
                batch,
                source_length,
                source_length,
                self.dim,
                self.heads,
            )?;
            let projected = matmul(
                &attended,
                &layer.attention.wo,
                rows,
                self.dim,
                Some(&layer.attention.bo),
            )?;
            encoder = residual_layer_norm(
                &projected,
                &encoder,
                &layer.attention.norm_scale,
                &layer.attention.norm_bias,
                rows,
                self.dim,
            )?;
            encoder = self.feed_forward(&encoder, &layer.ffn, rows)?;
        }

        let rows = batch * source_length;
        let mut cross = Vec::with_capacity(self.model.decoder.len());
        for layer in &self.model.decoder {
            let key = matmul(
                &encoder,
                &layer.context.wk,
                rows,
                self.dim,
                Some(&layer.context.bk),
            )?;
            let value = matmul(
                &encoder,
                &layer.context.wv,
                rows,
                self.dim,
                Some(&layer.context.bv),
            )?;
            cross.push(CrossCache { key, value });
        }
        Ok(EncodedBatch {
            source_length,
            lengths,
            cross,
        })
    }

    fn prepare_candidates(&self, tokens: &[i32], offsets: &[u32]) -> Result<Vec<Vec<u32>>, String> {
        offsets
            .windows(2)
            .map(|pair| {
                self.shortlist
                    .candidates(&tokens[pair[0] as usize..pair[1] as usize])
            })
            .collect()
    }

    fn embedding(&self, tokens: &[i32], batch: usize, sequence: usize) -> Result<Vec<f32>, String> {
        if tokens.len() != batch * sequence {
            return Err("embedding token shape does not match batch x sequence".into());
        }
        let mut output = vec![0.0_f32; batch * sequence * self.dim];
        for (token_offset, &token) in tokens.iter().enumerate() {
            let token =
                usize::try_from(token).map_err(|_| "embedding token ID is negative".to_string())?;
            if token >= self.model.encoder_embedding.rows() {
                return Err(format!(
                    "embedding token {token} exceeds vocabulary {}",
                    self.model.encoder_embedding.rows()
                ));
            }
            let position = token_offset % sequence;
            let source =
                &self.model.encoder_embedding.values()[token * self.dim..(token + 1) * self.dim];
            let position_values = &self.positions[position * self.dim..(position + 1) * self.dim];
            let destination = &mut output[token_offset * self.dim..(token_offset + 1) * self.dim];
            for ((value, &embedded), &positional) in
                destination.iter_mut().zip(source).zip(position_values)
            {
                *value = embedded * EMBEDDING_SCALE + positional;
            }
        }
        Ok(output)
    }

    fn decoder_input(
        &self,
        previous: &[i32],
        batch: usize,
        position: usize,
    ) -> Result<Vec<f32>, String> {
        if previous.len() != batch || position >= MAXIMUM_POSITION {
            return Err("decoder input shape or position is invalid".into());
        }
        let position_values = &self.positions[position * self.dim..(position + 1) * self.dim];
        let mut output = vec![0.0_f32; batch * self.dim];
        for row in 0..batch {
            let destination = &mut output[row * self.dim..(row + 1) * self.dim];
            if previous[row] < 0 {
                destination.copy_from_slice(position_values);
                continue;
            }
            let token = previous[row] as usize;
            if token >= self.model.decoder_embedding.rows() {
                return Err(format!(
                    "decoder token {token} exceeds vocabulary {}",
                    self.model.decoder_embedding.rows()
                ));
            }
            let embedding =
                &self.model.decoder_embedding.values()[token * self.dim..(token + 1) * self.dim];
            for ((value, &embedded), &positional) in
                destination.iter_mut().zip(embedding).zip(position_values)
            {
                *value = embedded * EMBEDDING_SCALE + positional;
            }
        }
        Ok(output)
    }

    fn feed_forward(
        &self,
        input: &[f32],
        weights: &FeedForwardWeights,
        rows: usize,
    ) -> Result<Vec<f32>, String> {
        let mut hidden = matmul(input, &weights.w1, rows, self.dim, Some(&weights.b1))?;
        relu_in_place(&mut hidden);
        let output = matmul(&hidden, &weights.w2, rows, self.ffn_dim, Some(&weights.b2))?;
        residual_layer_norm(
            &output,
            input,
            &weights.norm_scale,
            &weights.norm_bias,
            rows,
            self.dim,
        )
    }
}

impl ModelWeights {
    fn load(path: &Path, architecture: &Architecture) -> Result<Self, String> {
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
        // SAFETY: Read-only mapping; all tensors are copied into owned Matrix
        // storage before this function returns and the mapping is dropped.
        let bytes = unsafe { MmapOptions::new().map(&file) }
            .map_err(|error| format!("failed to map weights {}: {error}", path.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)
            .map_err(|error| format!("invalid safetensors {}: {error}", path.display()))?;
        let mut tensors = tensors.tensors().into_iter().collect::<HashMap<_, _>>();
        let dim = architecture.model_dim;
        let ffn_dim = architecture.ffn_dim;

        let encoder_embedding = take_tensor(
            &mut tensors,
            "encoder_Wemb",
            &[architecture.source_vocab_size, dim],
        )?;
        let decoder_embedding = take_tensor(
            &mut tensors,
            "decoder_Wemb",
            &[architecture.target_vocab_size, dim],
        )?;
        let output_bias = take_tensor(
            &mut tensors,
            "decoder_ff_logit_out_b",
            &[1, architecture.target_vocab_size],
        )?;
        let mut encoder = Vec::with_capacity(architecture.encoder_layers);
        for layer in 1..=architecture.encoder_layers {
            encoder.push(EncoderLayer {
                attention: load_attention(&mut tensors, &format!("encoder_l{layer}_self"), dim)?,
                ffn: load_ffn(&mut tensors, &format!("encoder_l{layer}_ffn"), dim, ffn_dim)?,
            });
        }
        let mut decoder = Vec::with_capacity(architecture.decoder_layers);
        for layer in 1..=architecture.decoder_layers {
            decoder.push(DecoderLayer {
                ssru: load_ssru(&mut tensors, &format!("decoder_l{layer}_rnn"), dim)?,
                context: load_attention(&mut tensors, &format!("decoder_l{layer}_context"), dim)?,
                ffn: load_ffn(&mut tensors, &format!("decoder_l{layer}_ffn"), dim, ffn_dim)?,
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
    tensors: &mut HashMap<String, TensorView<'_>>,
    prefix: &str,
    dim: usize,
) -> Result<AttentionWeights, String> {
    Ok(AttentionWeights {
        wq: take_tensor(tensors, &format!("{prefix}_Wq"), &[dim, dim])?,
        wk: take_tensor(tensors, &format!("{prefix}_Wk"), &[dim, dim])?,
        wv: take_tensor(tensors, &format!("{prefix}_Wv"), &[dim, dim])?,
        wo: take_tensor(tensors, &format!("{prefix}_Wo"), &[dim, dim])?,
        bq: take_tensor(tensors, &format!("{prefix}_bq"), &[1, dim])?,
        bk: take_tensor(tensors, &format!("{prefix}_bk"), &[1, dim])?,
        bv: take_tensor(tensors, &format!("{prefix}_bv"), &[1, dim])?,
        bo: take_tensor(tensors, &format!("{prefix}_bo"), &[1, dim])?,
        norm_scale: take_tensor(tensors, &format!("{prefix}_Wo_ln_scale"), &[1, dim])?,
        norm_bias: take_tensor(tensors, &format!("{prefix}_Wo_ln_bias"), &[1, dim])?,
    })
}

fn load_ffn(
    tensors: &mut HashMap<String, TensorView<'_>>,
    prefix: &str,
    dim: usize,
    ffn_dim: usize,
) -> Result<FeedForwardWeights, String> {
    Ok(FeedForwardWeights {
        w1: take_tensor(tensors, &format!("{prefix}_W1"), &[dim, ffn_dim])?,
        w2: take_tensor(tensors, &format!("{prefix}_W2"), &[ffn_dim, dim])?,
        b1: take_tensor(tensors, &format!("{prefix}_b1"), &[1, ffn_dim])?,
        b2: take_tensor(tensors, &format!("{prefix}_b2"), &[1, dim])?,
        norm_scale: take_tensor(tensors, &format!("{prefix}_ffn_ln_scale"), &[1, dim])?,
        norm_bias: take_tensor(tensors, &format!("{prefix}_ffn_ln_bias"), &[1, dim])?,
    })
}

fn load_ssru(
    tensors: &mut HashMap<String, TensorView<'_>>,
    prefix: &str,
    dim: usize,
) -> Result<SsruWeights, String> {
    Ok(SsruWeights {
        w: take_tensor(tensors, &format!("{prefix}_W"), &[dim, dim])?,
        wf: take_tensor(tensors, &format!("{prefix}_Wf"), &[dim, dim])?,
        bf: take_tensor(tensors, &format!("{prefix}_bf"), &[1, dim])?,
        norm_scale: take_tensor(tensors, &format!("{prefix}_ffn_ln_scale"), &[1, dim])?,
        norm_bias: take_tensor(tensors, &format!("{prefix}_ffn_ln_bias"), &[1, dim])?,
    })
}

fn take_tensor(
    tensors: &mut HashMap<String, TensorView<'_>>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Matrix, String> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| format!("missing model tensor {name}"))?;
    if tensor.dtype() != Dtype::F32 {
        return Err(format!(
            "model tensor {name} has dtype {:?}; pure Rust CPU v1 requires F32",
            tensor.dtype()
        ));
    }
    if tensor.shape() != expected_shape {
        return Err(format!(
            "model tensor {name} has shape {:?}; expected {expected_shape:?}",
            tensor.shape()
        ));
    }
    let values = decode_f32(tensor.data(), name)?;
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "model tensor {name} contains non-finite value {value} at element {index}"
        ));
    }
    Matrix::new(values, expected_shape[0], expected_shape[1])
}

fn validate_work_shape(batch: usize, source_length: usize) -> Result<(), String> {
    if source_length > MAXIMUM_SOURCE_LENGTH {
        return Err(format!(
            "source has {source_length} tokens; pure Rust FP32 maximum is {MAXIMUM_SOURCE_LENGTH}"
        ));
    }
    let attention_cells = batch
        .checked_mul(source_length)
        .and_then(|value| value.checked_mul(source_length))
        .ok_or_else(|| "padded attention work overflows usize".to_string())?;
    if attention_cells > MAXIMUM_PADDED_ATTENTION_CELLS {
        return Err(format!(
            "padded attention work is {attention_cells} cells; pure Rust FP32 maximum is {MAXIMUM_PADDED_ATTENTION_CELLS}"
        ));
    }
    Ok(())
}

#[cfg(target_endian = "little")]
fn decode_f32(bytes: &[u8], name: &str) -> Result<Vec<f32>, String> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(format!("model tensor {name} has a truncated F32 payload"));
    }
    let elements = bytes.len() / std::mem::size_of::<f32>();
    let mut values = vec![0.0_f32; elements];
    // SAFETY: f32 accepts every bit pattern, destination has exactly
    // bytes.len() writable bytes, and the allocations do not overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            values.as_mut_ptr().cast::<u8>(),
            bytes.len(),
        );
    }
    Ok(values)
}

#[cfg(target_endian = "big")]
fn decode_f32(bytes: &[u8], name: &str) -> Result<Vec<f32>, String> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(format!("model tensor {name} has a truncated F32 payload"));
    }
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let encoded: [u8; 4] = chunk
                .try_into()
                .map_err(|_| format!("model tensor {name} has invalid F32 data"))?;
            Ok(f32::from_le_bytes(encoded))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::validate_work_shape;

    #[test]
    fn padded_attention_work_is_bounded() {
        assert!(validate_work_shape(1, 256).is_ok());
        assert!(validate_work_shape(4, 128).is_ok());
        assert!(validate_work_shape(16, 64).is_ok());
        assert!(validate_work_shape(1, 257).is_err());
        assert!(validate_work_shape(2, 256).is_err());
    }
}
