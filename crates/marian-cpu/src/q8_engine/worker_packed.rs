use std::sync::Arc;

use marian_model::Architecture;

use super::{
    Matrix, Q8AttentionWeights, Q8DecoderLayer, Q8Embedding, Q8EncoderLayer, Q8FeedForwardWeights,
    Q8Linear, Q8ModelWeights, Q8SsruWeights,
};

const METADATA_MAGIC: &[u8; 8] = b"MARIWPK2";
const BUNDLE_MAGIC: &[u8; 8] = b"MARIBND2";
const VERSION: u32 = 2;
const WASM_KERNEL: &str = "wasm-u8i8i32";
const MAX_STRING_BYTES: usize = 256;
const BUNDLE_HEADER_BYTES: usize = 8 + 4 + 4 * 4;
type BundleSections<'a> = (&'a [u8], &'a [u8], &'a [u8], &'a [u8]);

/// Build one transport bundle. At runtime JS splits this into four ownership
/// sections so embeddings and packed dense weights are never copied in Wasm.
pub(super) fn encode(
    model: &Q8ModelWeights,
    architecture: &Architecture,
) -> Result<Vec<u8>, String> {
    let mut writer = Writer::default();
    writer.bytes.extend_from_slice(METADATA_MAGIC);
    writer.u32(VERSION);
    writer.string(WASM_KERNEL)?;
    writer.architecture(architecture)?;
    writer.embedding_header(&model.encoder_embedding)?;
    writer.embedding_header(&model.decoder_embedding)?;
    writer.f32(model.decoder_output_activation_mult)?;
    writer.f32_slice(&model.output_bias)?;
    for layer in &model.encoder {
        writer.attention(&layer.attention)?;
        writer.ffn(&layer.ffn)?;
    }
    for layer in &model.decoder {
        writer.ssru(&layer.ssru)?;
        writer.attention(&layer.context)?;
        writer.ffn(&layer.ffn)?;
    }

    let sections = [
        writer.bytes.len(),
        writer.dense.len(),
        model.encoder_embedding.values.len(),
        model.decoder_embedding.values.len(),
    ];
    let mut bundle = Vec::new();
    bundle.extend_from_slice(BUNDLE_MAGIC);
    bundle.extend_from_slice(&VERSION.to_le_bytes());
    for length in sections {
        bundle.extend_from_slice(
            &u32::try_from(length)
                .map_err(|_| "Worker packed section exceeds u32".to_string())?
                .to_le_bytes(),
        );
    }
    bundle.extend_from_slice(&writer.bytes);
    for word in writer.dense {
        bundle.extend_from_slice(&word.to_le_bytes());
    }
    bundle.extend(
        model
            .encoder_embedding
            .values
            .iter()
            .map(|value| *value as u8),
    );
    bundle.extend(
        model
            .decoder_embedding
            .values
            .iter()
            .map(|value| *value as u8),
    );
    Ok(bundle)
}

/// Convenience decoder used by non-streaming callers and tests.
pub(super) fn decode(bytes: &[u8], architecture: &Architecture) -> Result<Q8ModelWeights, String> {
    let (metadata, dense_bytes, encoder_bytes, decoder_bytes) = split_bundle(bytes)?;
    let dense = dense_bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte chunk")))
        .collect();
    decode_parts(
        metadata,
        dense,
        encoder_bytes.iter().map(|value| *value as i8).collect(),
        decoder_bytes.iter().map(|value| *value as i8).collect(),
        architecture,
    )
}

pub(super) fn decode_parts(
    metadata: &[u8],
    dense: Vec<u32>,
    encoder_values: Vec<i8>,
    decoder_values: Vec<i8>,
    architecture: &Architecture,
) -> Result<Q8ModelWeights, String> {
    let mut reader = Reader::new(metadata, dense);
    if reader.take(METADATA_MAGIC.len())? != METADATA_MAGIC {
        return Err("Worker packed metadata has an invalid magic header".into());
    }
    let version = reader.u32()?;
    if version != VERSION {
        return Err(format!(
            "Worker packed ABI version {version} is unsupported; expected {VERSION}"
        ));
    }
    let kernel = reader.string()?;
    if kernel != WASM_KERNEL {
        return Err(format!(
            "Worker packed kernel {kernel:?} is unsupported; expected {WASM_KERNEL:?}"
        ));
    }
    reader.architecture(architecture)?;
    let dim = architecture.model_dim;
    let ffn_dim = architecture.ffn_dim;
    let encoder_embedding =
        reader.embedding(architecture.source_vocab_size, dim, encoder_values)?;
    let decoder_embedding =
        reader.embedding(architecture.target_vocab_size, dim, decoder_values)?;
    let decoder_output_activation_mult = reader.positive_f32("decoder output activation scale")?;
    let output_bias = reader.f32_vec(architecture.target_vocab_size, "output bias")?;

    let mut encoder = Vec::with_capacity(architecture.encoder_layers);
    for layer in 1..=architecture.encoder_layers {
        encoder.push(Q8EncoderLayer {
            attention: reader.attention(&format!("encoder_l{layer}_self"), dim)?,
            ffn: reader.ffn(&format!("encoder_l{layer}_ffn"), dim, ffn_dim)?,
        });
    }
    let mut decoder = Vec::with_capacity(architecture.decoder_layers);
    for layer in 1..=architecture.decoder_layers {
        decoder.push(Q8DecoderLayer {
            ssru: reader.ssru(&format!("decoder_l{layer}_rnn"), dim)?,
            context: reader.attention(&format!("decoder_l{layer}_context"), dim)?,
            ffn: reader.ffn(&format!("decoder_l{layer}_ffn"), dim, ffn_dim)?,
        });
    }
    if !reader.remaining().is_empty() {
        return Err(format!(
            "Worker packed metadata has {} trailing bytes",
            reader.remaining().len()
        ));
    }
    Ok(Q8ModelWeights {
        encoder_embedding,
        decoder_embedding,
        decoder_output_activation_mult,
        output_bias,
        encoder,
        decoder,
    })
}

fn split_bundle(bytes: &[u8]) -> Result<BundleSections<'_>, String> {
    if bytes.len() < BUNDLE_HEADER_BYTES || &bytes[..8] != BUNDLE_MAGIC {
        return Err("Worker packed bundle has an invalid magic header".into());
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("header length"));
    if version != VERSION {
        return Err(format!(
            "Worker packed bundle version {version} is unsupported"
        ));
    }
    let mut lengths = [0_usize; 4];
    for (index, length) in lengths.iter_mut().enumerate() {
        let start = 12 + index * 4;
        *length =
            u32::from_le_bytes(bytes[start..start + 4].try_into().expect("header length")) as usize;
    }
    lengths[1] = lengths[1]
        .checked_mul(4)
        .ok_or_else(|| "Worker packed dense section overflows".to_string())?;
    let mut offset = BUNDLE_HEADER_BYTES;
    let mut sections = [&[][..]; 4];
    for (section, length) in sections.iter_mut().zip(lengths) {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| "Worker packed bundle offset overflows".to_string())?;
        *section = bytes
            .get(offset..end)
            .ok_or_else(|| "Worker packed bundle is truncated".to_string())?;
        offset = end;
    }
    if offset != bytes.len() {
        return Err(format!(
            "Worker packed bundle has {} trailing bytes",
            bytes.len() - offset
        ));
    }
    Ok((sections[0], sections[1], sections[2], sections[3]))
}

#[derive(Default)]
struct Writer {
    bytes: Vec<u8>,
    dense: Vec<u32>,
}

impl Writer {
    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize, label: &str) -> Result<(), String> {
        self.u32(u32::try_from(value).map_err(|_| format!("{label} {value} exceeds u32"))?);
        Ok(())
    }

    fn f32(&mut self, value: f32) -> Result<(), String> {
        if !value.is_finite() {
            return Err("Worker packed model cannot contain non-finite floats".into());
        }
        self.u32(value.to_bits());
        Ok(())
    }

    fn string(&mut self, value: &str) -> Result<(), String> {
        if value.len() > MAX_STRING_BYTES {
            return Err(format!(
                "Worker packed string is too long: {} bytes",
                value.len()
            ));
        }
        self.usize(value.len(), "string length")?;
        self.bytes.extend_from_slice(value.as_bytes());
        Ok(())
    }

    fn f32_slice(&mut self, values: &[f32]) -> Result<(), String> {
        self.usize(values.len(), "float vector length")?;
        for &value in values {
            self.f32(value)?;
        }
        Ok(())
    }

    fn architecture(&mut self, architecture: &Architecture) -> Result<(), String> {
        for (label, value) in [
            ("model_dim", architecture.model_dim),
            ("attention_heads", architecture.attention_heads),
            ("encoder_layers", architecture.encoder_layers),
            ("decoder_layers", architecture.decoder_layers),
            ("ffn_dim", architecture.ffn_dim),
            ("source_vocab_size", architecture.source_vocab_size),
            ("target_vocab_size", architecture.target_vocab_size),
            ("max_length_factor", architecture.max_length_factor),
        ] {
            self.usize(value, label)?;
        }
        self.bytes
            .extend_from_slice(&architecture.eos_id.to_le_bytes());
        self.bytes
            .extend_from_slice(&architecture.unk_id.to_le_bytes());
        Ok(())
    }

    fn embedding_header(&mut self, embedding: &Q8Embedding) -> Result<(), String> {
        self.usize(embedding.rows, "embedding rows")?;
        self.usize(embedding.cols, "embedding columns")?;
        self.f32(embedding.quant_mult)?;
        self.usize(embedding.values.len(), "embedding length")
    }

    fn matrix(&mut self, matrix: &Matrix) -> Result<(), String> {
        self.usize(matrix.rows(), "matrix rows")?;
        self.usize(matrix.cols(), "matrix columns")?;
        self.f32_slice(matrix.values())
    }

    fn linear(&mut self, linear: &Q8Linear) -> Result<(), String> {
        if linear.kernel_name() != WASM_KERNEL {
            return Err(format!(
                "{} uses kernel {}; generate this artifact inside the wasm32 SIMD build ({WASM_KERNEL})",
                linear.name(),
                linear.kernel_name()
            ));
        }
        self.string(linear.name())?;
        self.usize(linear.input_dim(), "linear input dimension")?;
        self.usize(linear.output_dim(), "linear output dimension")?;
        self.f32(linear.activation_quant_mult())?;
        self.f32(linear.weight_quant_mult())?;
        match linear.bias() {
            Some(bias) => self.f32_slice(bias)?,
            None => self.u32(u32::MAX),
        }
        let packed = linear.packed_words();
        self.usize(self.dense.len(), "packed word offset")?;
        self.usize(packed.len(), "packed word count")?;
        self.dense.extend(packed);
        Ok(())
    }

    fn attention(&mut self, weights: &Q8AttentionWeights) -> Result<(), String> {
        self.linear(&weights.wq)?;
        self.linear(&weights.wk)?;
        self.linear(&weights.wv)?;
        self.linear(&weights.wo)?;
        self.matrix(&weights.norm_scale)?;
        self.matrix(&weights.norm_bias)
    }

    fn ffn(&mut self, weights: &Q8FeedForwardWeights) -> Result<(), String> {
        self.linear(&weights.w1)?;
        self.linear(&weights.w2)?;
        self.matrix(&weights.norm_scale)?;
        self.matrix(&weights.norm_bias)
    }

    fn ssru(&mut self, weights: &Q8SsruWeights) -> Result<(), String> {
        self.linear(&weights.w)?;
        self.linear(&weights.wf)?;
        self.matrix(&weights.norm_scale)?;
        self.matrix(&weights.norm_bias)
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
    dense: Arc<Vec<u32>>,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8], dense: Vec<u32>) -> Self {
        Self {
            bytes,
            offset: 0,
            dense: Arc::new(dense),
        }
    }

    fn remaining(&self) -> &'a [u8] {
        &self.bytes[self.offset..]
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], String> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| "Worker packed metadata offset overflow".to_string())?;
        let value = self.bytes.get(self.offset..end).ok_or_else(|| {
            format!(
                "Worker packed metadata is truncated at byte {} (need {length})",
                self.offset
            )
        })?;
        self.offset = end;
        Ok(value)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four-byte read"),
        ))
    }

    fn usize(&mut self) -> Result<usize, String> {
        Ok(self.u32()? as usize)
    }

    fn f32(&mut self, label: &str) -> Result<f32, String> {
        let value = f32::from_bits(self.u32()?);
        if !value.is_finite() {
            return Err(format!("Worker packed {label} is non-finite"));
        }
        Ok(value)
    }

    fn positive_f32(&mut self, label: &str) -> Result<f32, String> {
        let value = self.f32(label)?;
        if value <= 0.0 {
            return Err(format!("Worker packed {label} must be positive"));
        }
        Ok(value)
    }

    fn string(&mut self) -> Result<String, String> {
        let length = self.usize()?;
        if length > MAX_STRING_BYTES {
            return Err(format!(
                "Worker packed string length {length} exceeds limit"
            ));
        }
        std::str::from_utf8(self.take(length)?)
            .map(str::to_owned)
            .map_err(|error| format!("Worker packed string is not UTF-8: {error}"))
    }

    fn architecture(&mut self, expected: &Architecture) -> Result<(), String> {
        let actual = [
            self.usize()?,
            self.usize()?,
            self.usize()?,
            self.usize()?,
            self.usize()?,
            self.usize()?,
            self.usize()?,
            self.usize()?,
        ];
        let expected_values = [
            expected.model_dim,
            expected.attention_heads,
            expected.encoder_layers,
            expected.decoder_layers,
            expected.ffn_dim,
            expected.source_vocab_size,
            expected.target_vocab_size,
            expected.max_length_factor,
        ];
        let eos_id = i32::from_le_bytes(self.take(4)?.try_into().expect("four-byte read"));
        let unk_id = i32::from_le_bytes(self.take(4)?.try_into().expect("four-byte read"));
        if actual != expected_values || eos_id != expected.eos_id || unk_id != expected.unk_id {
            return Err("Worker packed model architecture does not match manifest".into());
        }
        Ok(())
    }

    fn f32_vec(&mut self, expected: usize, label: &str) -> Result<Vec<f32>, String> {
        let length = self.usize()?;
        if length != expected {
            return Err(format!(
                "Worker packed {label} has {length} floats; expected {expected}"
            ));
        }
        (0..length).map(|_| self.f32(label)).collect()
    }

    fn embedding(
        &mut self,
        rows: usize,
        cols: usize,
        values: Vec<i8>,
    ) -> Result<Q8Embedding, String> {
        let actual_rows = self.usize()?;
        let actual_cols = self.usize()?;
        if (actual_rows, actual_cols) != (rows, cols) {
            return Err(format!(
                "Worker packed embedding is {actual_rows} x {actual_cols}; expected {rows} x {cols}"
            ));
        }
        let quant_mult = self.positive_f32("embedding quantization scale")?;
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| "embedding shape overflow".to_string())?;
        let length = self.usize()?;
        if length != expected || values.len() != expected {
            return Err(format!(
                "Worker packed embedding section has {} values; expected {expected}",
                values.len()
            ));
        }
        Ok(Q8Embedding {
            values,
            quant_mult,
            rows,
            cols,
        })
    }

    fn matrix(&mut self, rows: usize, cols: usize, label: &str) -> Result<Matrix, String> {
        let actual_rows = self.usize()?;
        let actual_cols = self.usize()?;
        if (actual_rows, actual_cols) != (rows, cols) {
            return Err(format!(
                "Worker packed {label} is {actual_rows} x {actual_cols}; expected {rows} x {cols}"
            ));
        }
        Matrix::new(self.f32_vec(rows * cols, label)?, rows, cols)
    }

    fn linear(&mut self, name: &str, input: usize, output: usize) -> Result<Q8Linear, String> {
        let actual_name = self.string()?;
        if actual_name != name {
            return Err(format!(
                "Worker packed linear {actual_name:?} is out of order; expected {name:?}"
            ));
        }
        let actual_input = self.usize()?;
        let actual_output = self.usize()?;
        if (actual_input, actual_output) != (input, output) {
            return Err(format!(
                "Worker packed {name} is {actual_input} x {actual_output}; expected {input} x {output}"
            ));
        }
        let activation = self.positive_f32("linear activation scale")?;
        let weight = self.positive_f32("linear weight scale")?;
        let bias_length = self.u32()?;
        let bias = if bias_length == u32::MAX {
            None
        } else {
            let length = bias_length as usize;
            if length != output {
                return Err(format!(
                    "Worker packed {name} bias has {length} values; expected {output}"
                ));
            }
            Some(
                (0..length)
                    .map(|_| self.f32("linear bias"))
                    .collect::<Result<_, _>>()?,
            )
        };
        let start = self.usize()?;
        let words = self.usize()?;
        Q8Linear::from_shared_packed_parts(
            name,
            input,
            output,
            activation,
            weight,
            bias,
            WASM_KERNEL,
            Arc::clone(&self.dense),
            start,
            words,
        )
        .map_err(|error| error.to_string())
    }

    fn attention(&mut self, prefix: &str, dim: usize) -> Result<Q8AttentionWeights, String> {
        Ok(Q8AttentionWeights {
            wq: self.linear(&format!("{prefix}_Wq"), dim, dim)?,
            wk: self.linear(&format!("{prefix}_Wk"), dim, dim)?,
            wv: self.linear(&format!("{prefix}_Wv"), dim, dim)?,
            wo: self.linear(&format!("{prefix}_Wo"), dim, dim)?,
            norm_scale: self.matrix(1, dim, "attention norm scale")?,
            norm_bias: self.matrix(1, dim, "attention norm bias")?,
        })
    }

    fn ffn(
        &mut self,
        prefix: &str,
        dim: usize,
        ffn_dim: usize,
    ) -> Result<Q8FeedForwardWeights, String> {
        Ok(Q8FeedForwardWeights {
            w1: self.linear(&format!("{prefix}_W1"), dim, ffn_dim)?,
            w2: self.linear(&format!("{prefix}_W2"), ffn_dim, dim)?,
            norm_scale: self.matrix(1, dim, "FFN norm scale")?,
            norm_bias: self.matrix(1, dim, "FFN norm bias")?,
        })
    }

    fn ssru(&mut self, prefix: &str, dim: usize) -> Result<Q8SsruWeights, String> {
        Ok(Q8SsruWeights {
            w: self.linear(&format!("{prefix}_W"), dim, dim)?,
            wf: self.linear(&format!("{prefix}_Wf"), dim, dim)?,
            norm_scale: self.matrix(1, dim, "SSRU norm scale")?,
            norm_bias: self.matrix(1, dim, "SSRU norm bias")?,
        })
    }
}
