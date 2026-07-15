use std::{cell::RefCell, collections::HashSet, path::Path, time::Instant};

use marian_model::{Architecture, LexicalShortlist};

use crate::{
    MarianBinaryModel, MarianTensorData, Q8Error, Q8Linear, Q8LinearScratch,
    quantize_symmetric_u8_into,
    tensor::{
        Matrix, attention_into, relu_in_place, residual_layer_norm_into,
        ssru_update_layer_norm_into,
    },
};

const MAXIMUM_POSITION: usize = 4_096;
const MAXIMUM_BATCH: usize = 256;
const MAXIMUM_SOURCE_LENGTH: usize = 256;
const MAXIMUM_GENERATION_STEPS: usize = 256;
const MAXIMUM_PADDED_ATTENTION_CELLS: usize = 65_536;
const EMBEDDING_SCALE: f32 = 19.595_919;

struct Q8Embedding {
    values: Vec<i8>,
    quant_mult: f32,
    rows: usize,
    cols: usize,
}

impl Q8Embedding {
    fn load(
        model: &MarianBinaryModel,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        let tensor = model.tensor(name).map_err(q8_string)?;
        if tensor.shape() != [rows, cols] {
            return Err(format!(
                "Q8 embedding {name} has shape {:?}; expected [{rows}, {cols}]",
                tensor.shape()
            ));
        }
        let (values, quant_mult) = tensor.as_intgemm8().map_err(q8_string)?;
        Ok(Self {
            values: values.to_vec(),
            quant_mult,
            rows,
            cols,
        })
    }

    fn row_into(&self, row: usize, destination: &mut [f32]) -> Result<(), String> {
        if row >= self.rows {
            return Err(format!(
                "Q8 embedding row {row} exceeds vocabulary {}",
                self.rows
            ));
        }
        if destination.len() != self.cols {
            return Err(format!(
                "Q8 embedding destination has {} values; expected {}",
                destination.len(),
                self.cols
            ));
        }
        for (output, &value) in destination
            .iter_mut()
            .zip(&self.values[row * self.cols..(row + 1) * self.cols])
        {
            *output = f32::from(value) / self.quant_mult;
        }
        Ok(())
    }
}

struct Q8AttentionWeights {
    wq: Q8Linear,
    wk: Q8Linear,
    wv: Q8Linear,
    wo: Q8Linear,
    norm_scale: Matrix,
    norm_bias: Matrix,
}

struct Q8FeedForwardWeights {
    w1: Q8Linear,
    w2: Q8Linear,
    norm_scale: Matrix,
    norm_bias: Matrix,
}

struct Q8SsruWeights {
    w: Q8Linear,
    wf: Q8Linear,
    norm_scale: Matrix,
    norm_bias: Matrix,
}

struct Q8EncoderLayer {
    attention: Q8AttentionWeights,
    ffn: Q8FeedForwardWeights,
}

struct Q8DecoderLayer {
    ssru: Q8SsruWeights,
    context: Q8AttentionWeights,
    ffn: Q8FeedForwardWeights,
}

struct Q8ModelWeights {
    encoder_embedding: Q8Embedding,
    decoder_embedding: Q8Embedding,
    decoder_output_activation_mult: f32,
    output_bias: Vec<f32>,
    encoder: Vec<Q8EncoderLayer>,
    decoder: Vec<Q8DecoderLayer>,
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

/// Pure Rust Q8 executor for Marian's fixed 384d Transformer-SSRU graph.
///
/// Dense weights remain int8 and are executed through [`Q8Linear`]. Only the
/// requested embedding rows and f32 activations are materialized. The tied
/// output projection quantizes each decoder state once and scores only its
/// lexical-shortlist rows.
pub struct Q8CpuEngine {
    model: Q8ModelWeights,
    positions: Vec<f32>,
    shortlist: LexicalShortlist,
    dim: usize,
    heads: usize,
    source_vocab: usize,
    max_length_factor: usize,
    linear_scratch: RefCell<Q8LinearScratch>,
    attention_scores: RefCell<Vec<f32>>,
    shortlist_quantized: RefCell<Vec<u8>>,
    buffer_pool: RefCell<Vec<Vec<f32>>>,
    packed_weight_build_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Q8MemoryReport {
    pub canonical_weight_bytes: usize,
    pub packed_weight_bytes: usize,
    pub embedding_bytes: usize,
    pub packed_weight_build_ms: f64,
}

impl Q8CpuEngine {
    pub fn load(
        weights_path: impl AsRef<Path>,
        shortlist_path: Option<&Path>,
        architecture: &Architecture,
    ) -> Result<Self, String> {
        validate_architecture(architecture)?;
        let binary = MarianBinaryModel::open(weights_path).map_err(q8_string)?;
        validate_tensor_schema(&binary, architecture)?;
        let packing_started = Instant::now();
        let model = Q8ModelWeights::load(&binary, architecture)?;
        let packed_weight_build_ms = packing_started.elapsed().as_secs_f64() * 1_000.0;
        let shortlist = LexicalShortlist::load(
            shortlist_path,
            architecture.source_vocab_size,
            architecture.target_vocab_size,
        )?;
        Ok(Self {
            model,
            positions: make_positions(architecture.model_dim),
            shortlist,
            dim: architecture.model_dim,
            heads: architecture.attention_heads,
            source_vocab: architecture.source_vocab_size,
            max_length_factor: architecture.max_length_factor.max(1),
            linear_scratch: RefCell::new(Q8LinearScratch::default()),
            attention_scores: RefCell::new(Vec::new()),
            shortlist_quantized: RefCell::new(Vec::new()),
            buffer_pool: RefCell::new(Vec::new()),
            packed_weight_build_ms,
        })
    }

    pub fn memory_report(&self) -> Q8MemoryReport {
        let mut canonical_weight_bytes = 0;
        let mut packed_weight_bytes = 0;
        self.model.for_each_linear(|linear| {
            let (canonical, packed) = linear.weight_storage_bytes();
            canonical_weight_bytes += canonical;
            packed_weight_bytes += packed;
        });
        Q8MemoryReport {
            canonical_weight_bytes,
            packed_weight_bytes,
            embedding_bytes: self.model.encoder_embedding.values.len()
                + self.model.decoder_embedding.values.len(),
            packed_weight_build_ms: self.packed_weight_build_ms,
        }
    }

    pub fn warmup(&self) -> Result<(), String> {
        self.translate_token_ids(&[2, 0], &[0, 2], &[4]).map(|_| ())
    }

    pub fn translate_token_ids(
        &self,
        tokens: &[i32],
        offsets: &[u32],
        max_output_tokens: &[usize],
    ) -> Result<crate::BatchOutput, String> {
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
                let update = self.run(&layer.ssru.w, &decoder, batch)?;
                let forget = self.run(&layer.ssru.wf, &decoder, batch)?;
                let next = self.ssru(
                    &update,
                    &forget,
                    &mut states[index],
                    &decoder,
                    &layer.ssru.norm_scale,
                    &layer.ssru.norm_bias,
                    batch,
                )?;
                self.recycle(update);
                self.recycle(forget);
                self.recycle(decoder);
                decoder = next;

                let query = self.run(&layer.context.wq, &decoder, batch)?;
                let attended = self.attention(
                    &query,
                    &encoded.cross[index].key,
                    &encoded.cross[index].value,
                    &encoded.lengths,
                    batch,
                    1,
                    encoded.source_length,
                )?;
                self.recycle(query);
                let context = self.run(&layer.context.wo, &attended, batch)?;
                self.recycle(attended);
                let next = self.residual(
                    &context,
                    &decoder,
                    &layer.context.norm_scale,
                    &layer.context.norm_bias,
                    batch,
                )?;
                self.recycle(context);
                self.recycle(decoder);
                decoder = next;
                let next = self.feed_forward(&decoder, &layer.ffn, batch)?;
                self.recycle(decoder);
                decoder = next;
            }

            for row in 0..batch {
                if finished[row] {
                    continue;
                }
                let state = &decoder[row * self.dim..(row + 1) * self.dim];
                let token = self.select_token(state, &candidates[row])? as i32;
                previous[row] = token;
                generated[row].push(token);
                if token == 0 {
                    finished[row] = true;
                }
            }
            self.recycle(decoder);
        }

        let mut output = crate::BatchOutput {
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
            let query = self.run(&layer.attention.wq, &encoder, rows)?;
            let key = self.run(&layer.attention.wk, &encoder, rows)?;
            let value = self.run(&layer.attention.wv, &encoder, rows)?;
            let attended = self.attention(
                &query,
                &key,
                &value,
                &lengths,
                batch,
                source_length,
                source_length,
            )?;
            self.recycle(query);
            self.recycle(key);
            self.recycle(value);
            let projected = self.run(&layer.attention.wo, &attended, rows)?;
            self.recycle(attended);
            let next = self.residual(
                &projected,
                &encoder,
                &layer.attention.norm_scale,
                &layer.attention.norm_bias,
                rows,
            )?;
            self.recycle(projected);
            self.recycle(encoder);
            encoder = next;
            let next = self.feed_forward(&encoder, &layer.ffn, rows)?;
            self.recycle(encoder);
            encoder = next;
        }

        let rows = batch * source_length;
        let mut cross = Vec::with_capacity(self.model.decoder.len());
        for layer in &self.model.decoder {
            cross.push(CrossCache {
                key: self.run(&layer.context.wk, &encoder, rows)?,
                value: self.run(&layer.context.wv, &encoder, rows)?,
            });
        }
        self.recycle(encoder);
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
        let mut output = self.take_buffer(batch * sequence * self.dim);
        for (token_offset, &token) in tokens.iter().enumerate() {
            let token =
                usize::try_from(token).map_err(|_| "embedding token ID is negative".to_string())?;
            let position = token_offset % sequence;
            let position_values = &self.positions[position * self.dim..(position + 1) * self.dim];
            let destination = &mut output[token_offset * self.dim..(token_offset + 1) * self.dim];
            self.model.encoder_embedding.row_into(token, destination)?;
            for (value, &positional) in destination.iter_mut().zip(position_values) {
                *value = *value * EMBEDDING_SCALE + positional;
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
        let mut output = self.take_buffer(batch * self.dim);
        for row in 0..batch {
            let destination = &mut output[row * self.dim..(row + 1) * self.dim];
            if previous[row] < 0 {
                destination.copy_from_slice(position_values);
                continue;
            }
            self.model
                .decoder_embedding
                .row_into(previous[row] as usize, destination)?;
            for (value, &positional) in destination.iter_mut().zip(position_values) {
                *value = *value * EMBEDDING_SCALE + positional;
            }
        }
        Ok(output)
    }

    fn feed_forward(
        &self,
        input: &[f32],
        weights: &Q8FeedForwardWeights,
        rows: usize,
    ) -> Result<Vec<f32>, String> {
        let mut hidden = self.run(&weights.w1, input, rows)?;
        relu_in_place(&mut hidden);
        let output = self.run(&weights.w2, &hidden, rows)?;
        let normalized = self.residual(
            &output,
            input,
            &weights.norm_scale,
            &weights.norm_bias,
            rows,
        )?;
        self.recycle(hidden);
        self.recycle(output);
        Ok(normalized)
    }

    fn run(&self, linear: &Q8Linear, input: &[f32], rows: usize) -> Result<Vec<f32>, String> {
        let mut output = self.take_buffer(rows.saturating_mul(linear.output_dim()));
        linear
            .run_into(
                input,
                rows,
                &mut output,
                &mut self.linear_scratch.borrow_mut(),
            )
            .map_err(q8_string)?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        lengths: &[usize],
        batch: usize,
        query_length: usize,
        key_length: usize,
    ) -> Result<Vec<f32>, String> {
        let mut output = self.take_buffer(batch * query_length * self.dim);
        attention_into(
            query,
            key,
            value,
            lengths,
            batch,
            query_length,
            key_length,
            self.dim,
            self.heads,
            &mut output,
            &mut self.attention_scores.borrow_mut(),
        )?;
        Ok(output)
    }

    fn residual(
        &self,
        input: &[f32],
        residual: &[f32],
        scale: &Matrix,
        bias: &Matrix,
        rows: usize,
    ) -> Result<Vec<f32>, String> {
        let mut output = self.take_buffer(rows * self.dim);
        residual_layer_norm_into(input, residual, scale, bias, rows, self.dim, &mut output)?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn ssru(
        &self,
        candidate: &[f32],
        forget: &[f32],
        state: &mut [f32],
        residual: &[f32],
        scale: &Matrix,
        bias: &Matrix,
        rows: usize,
    ) -> Result<Vec<f32>, String> {
        let mut output = self.take_buffer(rows * self.dim);
        ssru_update_layer_norm_into(
            candidate,
            forget,
            state,
            residual,
            scale,
            bias,
            rows,
            self.dim,
            &mut output,
        )?;
        Ok(output)
    }

    fn take_buffer(&self, elements: usize) -> Vec<f32> {
        let mut pool = self.buffer_pool.borrow_mut();
        let mut buffer = pool.pop().unwrap_or_else(|| Vec::with_capacity(elements));
        buffer.clear();
        buffer.resize(elements, 0.0);
        buffer
    }

    fn recycle(&self, mut buffer: Vec<f32>) {
        buffer.clear();
        let mut pool = self.buffer_pool.borrow_mut();
        if pool.len() < 32 {
            pool.push(buffer);
        }
    }

    fn select_token(&self, decoder: &[f32], candidates: &[u32]) -> Result<u32, String> {
        if decoder.len() != self.dim {
            return Err(format!(
                "decoder state has {} elements, expected {}",
                decoder.len(),
                self.dim
            ));
        }
        if candidates.is_empty() {
            return Err("cannot select from an empty candidate list".into());
        }
        let mut quantized = self.shortlist_quantized.borrow_mut();
        quantize_symmetric_u8_into(
            decoder,
            self.model.decoder_output_activation_mult,
            &mut quantized,
        )
        .map_err(q8_string)?;
        let activation_mult = self.model.decoder_output_activation_mult;
        let weight_mult = self.model.decoder_embedding.quant_mult;
        let inverse_scale = 1.0 / (activation_mult * weight_mult);
        let mut best_token = candidates[0];
        let mut best_value = f32::NEG_INFINITY;
        for &candidate in candidates {
            let token = candidate as usize;
            if token >= self.model.decoder_embedding.rows {
                return Err(format!(
                    "output candidate {token} exceeds vocabulary {}",
                    self.model.decoder_embedding.rows
                ));
            }
            let weights =
                &self.model.decoder_embedding.values[token * self.dim..(token + 1) * self.dim];
            let accumulator = crate::q8_arm::dot_u8_i8(&quantized, weights);
            let logit = accumulator as f32 * inverse_scale + self.model.output_bias[token];
            if logit > best_value {
                best_value = logit;
                best_token = candidate;
            }
        }
        Ok(best_token)
    }
}

impl Q8ModelWeights {
    fn for_each_linear(&self, mut visit: impl FnMut(&Q8Linear)) {
        for layer in &self.encoder {
            for linear in [
                &layer.attention.wq,
                &layer.attention.wk,
                &layer.attention.wv,
                &layer.attention.wo,
                &layer.ffn.w1,
                &layer.ffn.w2,
            ] {
                visit(linear);
            }
        }
        for layer in &self.decoder {
            for linear in [
                &layer.ssru.w,
                &layer.ssru.wf,
                &layer.context.wq,
                &layer.context.wk,
                &layer.context.wv,
                &layer.context.wo,
                &layer.ffn.w1,
                &layer.ffn.w2,
            ] {
                visit(linear);
            }
        }
    }

    fn load(model: &MarianBinaryModel, architecture: &Architecture) -> Result<Self, String> {
        let dim = architecture.model_dim;
        let ffn_dim = architecture.ffn_dim;
        let encoder_embedding =
            Q8Embedding::load(model, "encoder_Wemb", architecture.source_vocab_size, dim)?;
        let decoder_embedding =
            Q8Embedding::load(model, "decoder_Wemb", architecture.target_vocab_size, dim)?;
        let decoder_output_activation_mult = model
            .activation_scale_for("decoder_Wemb")
            .map_err(q8_string)?;
        let output_bias = take_float_vector(
            model,
            "decoder_ff_logit_out_b",
            architecture.target_vocab_size,
        )?;
        let mut encoder = Vec::with_capacity(architecture.encoder_layers);
        for layer in 1..=architecture.encoder_layers {
            encoder.push(Q8EncoderLayer {
                attention: load_attention(model, &format!("encoder_l{layer}_self"), dim)?,
                ffn: load_ffn(model, &format!("encoder_l{layer}_ffn"), dim, ffn_dim)?,
            });
        }
        let mut decoder = Vec::with_capacity(architecture.decoder_layers);
        for layer in 1..=architecture.decoder_layers {
            decoder.push(Q8DecoderLayer {
                ssru: load_ssru(model, &format!("decoder_l{layer}_rnn"), dim)?,
                context: load_attention(model, &format!("decoder_l{layer}_context"), dim)?,
                ffn: load_ffn(model, &format!("decoder_l{layer}_ffn"), dim, ffn_dim)?,
            });
        }
        Ok(Self {
            encoder_embedding,
            decoder_embedding,
            decoder_output_activation_mult,
            output_bias,
            encoder,
            decoder,
        })
    }
}

fn load_attention(
    model: &MarianBinaryModel,
    prefix: &str,
    dim: usize,
) -> Result<Q8AttentionWeights, String> {
    Ok(Q8AttentionWeights {
        wq: dense(
            model,
            &format!("{prefix}_Wq"),
            Some(&format!("{prefix}_bq")),
            dim,
            dim,
        )?,
        wk: dense(
            model,
            &format!("{prefix}_Wk"),
            Some(&format!("{prefix}_bk")),
            dim,
            dim,
        )?,
        wv: dense(
            model,
            &format!("{prefix}_Wv"),
            Some(&format!("{prefix}_bv")),
            dim,
            dim,
        )?,
        wo: dense(
            model,
            &format!("{prefix}_Wo"),
            Some(&format!("{prefix}_bo")),
            dim,
            dim,
        )?,
        norm_scale: take_float_matrix(model, &format!("{prefix}_Wo_ln_scale"), 1, dim)?,
        norm_bias: take_float_matrix(model, &format!("{prefix}_Wo_ln_bias"), 1, dim)?,
    })
}

fn load_ffn(
    model: &MarianBinaryModel,
    prefix: &str,
    dim: usize,
    ffn_dim: usize,
) -> Result<Q8FeedForwardWeights, String> {
    Ok(Q8FeedForwardWeights {
        w1: dense(
            model,
            &format!("{prefix}_W1"),
            Some(&format!("{prefix}_b1")),
            dim,
            ffn_dim,
        )?,
        w2: dense(
            model,
            &format!("{prefix}_W2"),
            Some(&format!("{prefix}_b2")),
            ffn_dim,
            dim,
        )?,
        norm_scale: take_float_matrix(model, &format!("{prefix}_ffn_ln_scale"), 1, dim)?,
        norm_bias: take_float_matrix(model, &format!("{prefix}_ffn_ln_bias"), 1, dim)?,
    })
}

fn load_ssru(model: &MarianBinaryModel, prefix: &str, dim: usize) -> Result<Q8SsruWeights, String> {
    Ok(Q8SsruWeights {
        w: dense(model, &format!("{prefix}_W"), None, dim, dim)?,
        wf: dense(
            model,
            &format!("{prefix}_Wf"),
            Some(&format!("{prefix}_bf")),
            dim,
            dim,
        )?,
        norm_scale: take_float_matrix(model, &format!("{prefix}_ffn_ln_scale"), 1, dim)?,
        norm_bias: take_float_matrix(model, &format!("{prefix}_ffn_ln_bias"), 1, dim)?,
    })
}

fn dense(
    model: &MarianBinaryModel,
    weight: &str,
    bias: Option<&str>,
    input_dim: usize,
    output_dim: usize,
) -> Result<Q8Linear, String> {
    model
        .dense_linear(weight, bias, input_dim, output_dim)
        .map_err(q8_string)
}

fn take_float_matrix(
    model: &MarianBinaryModel,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Matrix, String> {
    let tensor = model.tensor(name).map_err(q8_string)?;
    if tensor.shape() != [rows, cols] && !(rows == 1 && tensor.shape() == [cols]) {
        return Err(format!(
            "Q8 model tensor {name} has shape {:?}; expected [{rows}, {cols}]",
            tensor.shape()
        ));
    }
    Matrix::new(tensor.as_float32().map_err(q8_string)?.to_vec(), rows, cols)
}

fn take_float_vector(
    model: &MarianBinaryModel,
    name: &str,
    elements: usize,
) -> Result<Vec<f32>, String> {
    let tensor = model.tensor(name).map_err(q8_string)?;
    if tensor.shape() != [1, elements] && tensor.shape() != [elements] {
        return Err(format!(
            "Q8 model tensor {name} has shape {:?}; expected [1, {elements}]",
            tensor.shape()
        ));
    }
    Ok(tensor.as_float32().map_err(q8_string)?.to_vec())
}

fn validate_tensor_schema(
    model: &MarianBinaryModel,
    architecture: &Architecture,
) -> Result<(), String> {
    let report = model.validate_q8().map_err(q8_string)?;
    if (
        report.dense_linears,
        report.embeddings,
        report.activation_scales,
    ) != (68, 2, 69)
    {
        return Err(format!(
            "Q8 tensor inventory is dense={} embeddings={} activation_scales={}; expected 68/2/69",
            report.dense_linears, report.embeddings, report.activation_scales
        ));
    }
    let expected = expected_tensor_names(architecture);
    let actual = model
        .tensors()
        .iter()
        .map(|tensor| tensor.name().to_owned())
        .collect::<HashSet<_>>();
    if actual != expected {
        let mut missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let mut unexpected = actual.difference(&expected).cloned().collect::<Vec<_>>();
        missing.sort();
        unexpected.sort();
        return Err(format!(
            "Q8 tensor schema mismatch; missing [{}], unexpected [{}]",
            missing.into_iter().take(8).collect::<Vec<_>>().join(", "),
            unexpected
                .into_iter()
                .take(8)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    match model.tensor("special:model.yml").map_err(q8_string)?.data() {
        MarianTensorData::Int8(_) => Ok(()),
        _ => Err("special:model.yml must be a raw int8 byte tensor".into()),
    }
}

fn expected_tensor_names(architecture: &Architecture) -> HashSet<String> {
    let mut names = HashSet::new();
    for name in [
        "encoder_Wemb",
        "decoder_Wemb",
        "decoder_Wemb_QuantMultA",
        "decoder_ff_logit_out_b",
        "special:model.yml",
    ] {
        names.insert(name.to_owned());
    }
    for layer in 1..=architecture.encoder_layers {
        add_attention_names(&mut names, &format!("encoder_l{layer}_self"));
        add_ffn_names(&mut names, &format!("encoder_l{layer}_ffn"));
    }
    for layer in 1..=architecture.decoder_layers {
        add_ssru_names(&mut names, &format!("decoder_l{layer}_rnn"));
        add_attention_names(&mut names, &format!("decoder_l{layer}_context"));
        add_ffn_names(&mut names, &format!("decoder_l{layer}_ffn"));
    }
    names
}

fn add_dense_name(names: &mut HashSet<String>, name: String) {
    names.insert(name.clone());
    names.insert(format!("{name}_QuantMultA"));
}

fn add_attention_names(names: &mut HashSet<String>, prefix: &str) {
    for suffix in ["Wq", "Wk", "Wv", "Wo"] {
        add_dense_name(names, format!("{prefix}_{suffix}"));
    }
    for suffix in ["bq", "bk", "bv", "bo", "Wo_ln_scale", "Wo_ln_bias"] {
        names.insert(format!("{prefix}_{suffix}"));
    }
}

fn add_ffn_names(names: &mut HashSet<String>, prefix: &str) {
    add_dense_name(names, format!("{prefix}_W1"));
    add_dense_name(names, format!("{prefix}_W2"));
    for suffix in ["b1", "b2", "ffn_ln_scale", "ffn_ln_bias"] {
        names.insert(format!("{prefix}_{suffix}"));
    }
}

fn add_ssru_names(names: &mut HashSet<String>, prefix: &str) {
    add_dense_name(names, format!("{prefix}_W"));
    add_dense_name(names, format!("{prefix}_Wf"));
    for suffix in ["bf", "ffn_ln_scale", "ffn_ln_bias"] {
        names.insert(format!("{prefix}_{suffix}"));
    }
}

fn validate_work_shape(batch: usize, source_length: usize) -> Result<(), String> {
    if source_length > MAXIMUM_SOURCE_LENGTH {
        return Err(format!(
            "source has {source_length} tokens; pure Rust Q8 maximum is {MAXIMUM_SOURCE_LENGTH}"
        ));
    }
    let attention_cells = batch
        .checked_mul(source_length)
        .and_then(|value| value.checked_mul(source_length))
        .ok_or_else(|| "padded attention work overflows usize".to_string())?;
    if attention_cells > MAXIMUM_PADDED_ATTENTION_CELLS {
        return Err(format!(
            "padded attention work is {attention_cells} cells; pure Rust Q8 maximum is {MAXIMUM_PADDED_ATTENTION_CELLS}"
        ));
    }
    Ok(())
}

fn validate_architecture(architecture: &Architecture) -> Result<(), String> {
    if (
        architecture.model_dim,
        architecture.attention_heads,
        architecture.encoder_layers,
        architecture.decoder_layers,
        architecture.ffn_dim,
    ) != (384, 8, 6, 4, 1536)
        || architecture.eos_id != 0
        || architecture.unk_id != 1
    {
        return Err("pure Rust Q8 supports the 384d/6e/4d SSRU graph only".into());
    }
    if architecture.source_vocab_size <= 2 || architecture.target_vocab_size <= 2 {
        return Err("model vocabulary sizes must contain EOS, UNK, and warmup tokens".into());
    }
    if !(1..=8).contains(&architecture.max_length_factor) {
        return Err("model max_length_factor must be between 1 and 8".into());
    }
    Ok(())
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

fn q8_string(error: Q8Error) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::expected_tensor_names;
    use marian_model::Architecture;

    #[test]
    fn fixed_graph_consumes_exactly_253_tensors() {
        let architecture = Architecture {
            model_dim: 384,
            attention_heads: 8,
            encoder_layers: 6,
            decoder_layers: 4,
            ffn_dim: 1536,
            source_vocab_size: 32_000,
            target_vocab_size: 32_000,
            eos_id: 0,
            unk_id: 1,
            max_length_factor: 2,
        };
        assert_eq!(expected_tensor_names(&architecture).len(), 253);
    }
}
