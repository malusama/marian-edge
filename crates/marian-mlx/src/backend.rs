use std::path::Path;

use marian_core::{
    BackendError, BackendInfo, TranslationBackend, TranslationInput, TranslationOutput,
};
use sentencepiece::SentencePieceProcessor;

use crate::{ffi::bridge, manifest::ModelManifest};

const MAX_SOURCE_TOKENS: usize = 4_096;

pub struct MlxBackend {
    engine: cxx::UniquePtr<bridge::Engine>,
    source: SentencePieceProcessor,
    target: SentencePieceProcessor,
    source_lang: String,
    target_lang: String,
    model_id: String,
    precision: String,
    eos_id: i32,
    device: String,
}

impl MlxBackend {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, BackendError> {
        let model_dir = std::fs::canonicalize(model_dir.as_ref()).map_err(|error| {
            BackendError::Model(format!(
                "failed to resolve model directory {}: {error}",
                model_dir.as_ref().display()
            ))
        })?;
        let manifest = ModelManifest::load(&model_dir)?;
        manifest.verify_runtime_files(&model_dir)?;
        let weights = absolute_utf8(&model_dir.join(&manifest.weights))?;
        let source_vocab = model_dir.join(&manifest.source_vocab);
        let target_vocab = model_dir.join(&manifest.target_vocab);
        let shortlist = manifest
            .shortlist
            .as_ref()
            .map(|path| absolute_utf8(&model_dir.join(path)))
            .transpose()?
            .unwrap_or_default();
        let metallib = std::env::var("MARIAN_MLX_METALLIB").unwrap_or_default();

        let source = SentencePieceProcessor::open(&source_vocab).map_err(|error| {
            BackendError::Model(format!(
                "failed to load source vocabulary {}: {error}",
                source_vocab.display()
            ))
        })?;
        let target = SentencePieceProcessor::open(&target_vocab).map_err(|error| {
            BackendError::Model(format!(
                "failed to load target vocabulary {}: {error}",
                target_vocab.display()
            ))
        })?;
        if source.len() != manifest.architecture.source_vocab_size
            || target.len() != manifest.architecture.target_vocab_size
        {
            return Err(BackendError::Model(
                "SentencePiece vocabulary size does not match manifest".into(),
            ));
        }

        let mut engine = bridge::new_engine(
            &weights,
            &shortlist,
            &metallib,
            manifest.architecture.max_length_factor,
        )
        .map_err(|error| BackendError::Model(error.to_string()))?;
        engine
            .pin_mut()
            .warmup()
            .map_err(|error| BackendError::Inference(format!("MLX warmup failed: {error}")))?;
        let device = engine
            .as_ref()
            .ok_or_else(|| BackendError::Model("MLX returned a null engine".into()))?
            .device_name();

        Ok(Self {
            engine,
            source,
            target,
            source_lang: manifest.source_lang,
            target_lang: manifest.target_lang,
            model_id: manifest.model_id,
            precision: manifest.precision,
            eos_id: manifest.architecture.eos_id,
            device,
        })
    }
}

impl TranslationBackend for MlxBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "mlx".into(),
            device: self.device.clone(),
            model: self.model_id.clone(),
            precision: self.precision.clone(),
            supports_batching: true,
        }
    }

    fn translate_batch(
        &mut self,
        inputs: &[TranslationInput],
    ) -> Result<Vec<TranslationOutput>, BackendError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let max_output_tokens = inputs[0].max_output_tokens;
        let mut tokens = Vec::new();
        let mut offsets = Vec::with_capacity(inputs.len() + 1);
        offsets.push(0);
        let mut input_lengths = Vec::with_capacity(inputs.len());

        for input in inputs {
            if input.source_lang != self.source_lang || input.target_lang != self.target_lang {
                return Err(BackendError::UnsupportedDirection(format!(
                    "{} -> {}; loaded model is {} -> {}",
                    input.source_lang, input.target_lang, self.source_lang, self.target_lang
                )));
            }
            let pieces = self.source.encode(&input.text).map_err(|error| {
                BackendError::InvalidInput(format!("SentencePiece encoding failed: {error}"))
            })?;
            if pieces.len() + 1 > MAX_SOURCE_TOKENS {
                return Err(BackendError::InvalidInput(format!(
                    "source has {} tokens including EOS; maximum is {MAX_SOURCE_TOKENS}",
                    pieces.len() + 1
                )));
            }
            tokens.extend(pieces.iter().map(|piece| piece.id as i32));
            tokens.push(self.eos_id);
            input_lengths.push(pieces.len() + 1);
            offsets.push(tokens.len() as u32);
        }

        let output = self
            .engine
            .pin_mut()
            .translate(&tokens, &offsets, max_output_tokens)
            .map_err(|error| BackendError::Inference(error.to_string()))?;
        if output.offsets.len() != inputs.len() + 1 || output.scores.len() != inputs.len() {
            return Err(BackendError::Inference(
                "MLX engine returned malformed batch offsets".into(),
            ));
        }

        let mut results = Vec::with_capacity(inputs.len());
        for (index, &input_tokens) in input_lengths.iter().enumerate() {
            let start = output.offsets[index] as usize;
            let end = output.offsets[index + 1] as usize;
            if end < start || end > output.tokens.len() {
                return Err(BackendError::Inference(
                    "MLX engine returned out-of-range token offsets".into(),
                ));
            }
            let ids: Vec<u32> = output.tokens[start..end]
                .iter()
                .copied()
                .filter(|id| *id != self.eos_id)
                .map(|id| id as u32)
                .collect();
            let text = self.target.decode_piece_ids(&ids).map_err(|error| {
                BackendError::Inference(format!("SentencePiece decoding failed: {error}"))
            })?;
            results.push(TranslationOutput {
                text,
                // The current greedy C++ decoder does not expose a calibrated
                // sequence score. Do not present its placeholder as real data.
                score: None,
                input_tokens,
                output_tokens: ids.len(),
            });
        }
        Ok(results)
    }
}

fn absolute_utf8(path: &Path) -> Result<String, BackendError> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| BackendError::Model(format!("path is not valid UTF-8: {}", path.display())))
}
