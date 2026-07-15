use std::{collections::HashMap, fs, mem::size_of, path::Path};

use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors, tensor::TensorView as SafeTensorView};

use marian_model::Architecture;

use super::{MAXIMUM_POSITION, checked_product};
use crate::metal_runtime::{Buffer, MetalRuntime};

pub(super) struct AttentionOutputWeights {
    pub(super) weight: Buffer,
    pub(super) bias: Buffer,
    pub(super) norm_scale: Buffer,
    pub(super) norm_bias: Buffer,
}

pub(super) struct SelfAttentionWeights {
    pub(super) qkv: Buffer,
    pub(super) qkv_bias: Buffer,
    pub(super) output: AttentionOutputWeights,
}

pub(super) struct CrossAttentionWeights {
    pub(super) query: Buffer,
    pub(super) query_bias: Buffer,
    pub(super) key_value: Buffer,
    pub(super) key_value_bias: Buffer,
    pub(super) output: AttentionOutputWeights,
}

pub(super) struct FeedForwardWeights {
    pub(super) w1: Buffer,
    pub(super) w2: Buffer,
    pub(super) b1: Buffer,
    pub(super) b2: Buffer,
    pub(super) norm_scale: Buffer,
    pub(super) norm_bias: Buffer,
}

pub(super) struct SsruWeights {
    pub(super) projection: Buffer,
    pub(super) bf: Buffer,
    pub(super) norm_scale: Buffer,
    pub(super) norm_bias: Buffer,
}

pub(super) struct EncoderLayer {
    pub(super) attention: SelfAttentionWeights,
    pub(super) ffn: FeedForwardWeights,
}

pub(super) struct DecoderLayer {
    pub(super) ssru: SsruWeights,
    pub(super) context: CrossAttentionWeights,
    pub(super) ffn: FeedForwardWeights,
}

pub(super) struct ModelWeights {
    pub(super) encoder_embedding: Buffer,
    pub(super) decoder_embedding: Buffer,
    pub(super) output_bias: Buffer,
    pub(super) encoder: Vec<EncoderLayer>,
    pub(super) decoder: Vec<DecoderLayer>,
}

impl ModelWeights {
    pub(super) fn load(
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
                attention: load_self_attention(
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
                context: load_cross_attention(
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

fn load_self_attention(
    runtime: &MetalRuntime,
    tensors: &mut HashMap<String, SafeTensorView<'_>>,
    prefix: &str,
    dim: usize,
) -> Result<SelfAttentionWeights, String> {
    let wq = format!("{prefix}_Wq");
    let wk = format!("{prefix}_Wk");
    let wv = format!("{prefix}_Wv");
    let bq = format!("{prefix}_bq");
    let bk = format!("{prefix}_bk");
    let bv = format!("{prefix}_bv");
    Ok(SelfAttentionWeights {
        qkv: take_packed_columns(runtime, tensors, &[&wq, &wk, &wv], dim, dim)?,
        qkv_bias: take_packed_columns(runtime, tensors, &[&bq, &bk, &bv], 1, dim)?,
        output: load_attention_output(runtime, tensors, prefix, dim)?,
    })
}

fn load_cross_attention(
    runtime: &MetalRuntime,
    tensors: &mut HashMap<String, SafeTensorView<'_>>,
    prefix: &str,
    dim: usize,
) -> Result<CrossAttentionWeights, String> {
    let wk = format!("{prefix}_Wk");
    let wv = format!("{prefix}_Wv");
    let bk = format!("{prefix}_bk");
    let bv = format!("{prefix}_bv");
    Ok(CrossAttentionWeights {
        query: take_tensor(runtime, tensors, &format!("{prefix}_Wq"), &[dim, dim])?,
        query_bias: take_tensor(runtime, tensors, &format!("{prefix}_bq"), &[1, dim])?,
        key_value: take_packed_columns(runtime, tensors, &[&wk, &wv], dim, dim)?,
        key_value_bias: take_packed_columns(runtime, tensors, &[&bk, &bv], 1, dim)?,
        output: load_attention_output(runtime, tensors, prefix, dim)?,
    })
}

fn load_attention_output(
    runtime: &MetalRuntime,
    tensors: &mut HashMap<String, SafeTensorView<'_>>,
    prefix: &str,
    dim: usize,
) -> Result<AttentionOutputWeights, String> {
    Ok(AttentionOutputWeights {
        weight: take_tensor(runtime, tensors, &format!("{prefix}_Wo"), &[dim, dim])?,
        bias: take_tensor(runtime, tensors, &format!("{prefix}_bo"), &[1, dim])?,
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
    tensors: &mut HashMap<String, SafeTensorView<'_>>,
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
    tensors: &mut HashMap<String, SafeTensorView<'_>>,
    prefix: &str,
    dim: usize,
) -> Result<SsruWeights, String> {
    Ok(SsruWeights {
        projection: take_packed_columns(
            runtime,
            tensors,
            &[&format!("{prefix}_W"), &format!("{prefix}_Wf")],
            dim,
            dim,
        )?,
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
    tensors: &mut HashMap<String, SafeTensorView<'_>>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Buffer, String> {
    let tensor = remove_tensor(tensors, name, expected_shape)?;
    runtime.upload_model_f32(tensor.data())
}

fn remove_tensor<'a>(
    tensors: &mut HashMap<String, SafeTensorView<'a>>,
    name: &str,
    expected_shape: &[usize],
) -> Result<SafeTensorView<'a>, String> {
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
    Ok(tensor)
}

fn take_packed_columns(
    runtime: &MetalRuntime,
    tensors: &mut HashMap<String, SafeTensorView<'_>>,
    names: &[&str],
    rows: usize,
    columns: usize,
) -> Result<Buffer, String> {
    if names.len() < 2 {
        return Err("packed model projection requires at least two tensors".into());
    }
    let tensors = names
        .iter()
        .map(|name| remove_tensor(tensors, name, &[rows, columns]))
        .collect::<Result<Vec<_>, _>>()?;
    let row_bytes = checked_product(&[columns, size_of::<f32>()])?;
    let mut packed = Vec::with_capacity(checked_product(&[rows, row_bytes, names.len()])?);
    for row in 0..rows {
        let start = row
            .checked_mul(row_bytes)
            .ok_or_else(|| "packed model tensor row offset overflows usize".to_string())?;
        let end = start + row_bytes;
        for tensor in &tensors {
            packed.extend_from_slice(&tensor.data()[start..end]);
        }
    }
    runtime.upload_model_f32(&packed)
}

pub(super) fn make_positions(dim: usize) -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::make_positions;

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
