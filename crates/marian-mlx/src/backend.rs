use std::path::Path;

use marian_core::{
    BackendError, BackendInfo, TranslationBackend, TranslationInput, TranslationOutput,
};
use marian_model::ModelManifest;
use marian_tokenizer::Tokenizer;

use crate::engine::MetalEngine;

const MAX_SOURCE_TOKENS: usize = 4_096;

pub struct MetalBackend {
    engine: MetalEngine,
    source: Tokenizer,
    target: Tokenizer,
    source_lang: String,
    target_lang: String,
    model_id: String,
    precision: String,
    eos_id: i32,
    device: String,
}

impl MetalBackend {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, BackendError> {
        let model_dir = std::fs::canonicalize(model_dir.as_ref()).map_err(|error| {
            BackendError::Model(format!(
                "failed to resolve model directory {}: {error}",
                model_dir.as_ref().display()
            ))
        })?;
        let manifest = ModelManifest::load(&model_dir)?;
        if manifest.precision != "fp32" {
            return Err(BackendError::Model(format!(
                "Metal requires fp32 safetensors weights, got {}",
                manifest.precision
            )));
        }
        manifest.verify_runtime_files(&model_dir)?;
        let weights = model_dir.join(&manifest.weights);
        let source_vocab = model_dir.join(&manifest.source_vocab);
        let target_vocab = model_dir.join(&manifest.target_vocab);
        let shortlist = manifest.shortlist.as_ref().map(|path| model_dir.join(path));

        let source = Tokenizer::open(&source_vocab).map_err(|error| {
            BackendError::Model(format!(
                "failed to load source vocabulary {}: {error}",
                source_vocab.display()
            ))
        })?;
        let target = Tokenizer::open(&target_vocab).map_err(|error| {
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

        let engine = MetalEngine::load(&weights, shortlist.as_deref(), &manifest.architecture)
            .map_err(BackendError::Model)?;
        engine
            .warmup()
            .map_err(|error| BackendError::Inference(format!("Metal warmup failed: {error}")))?;
        let device = engine.device_name().to_owned();

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

impl TranslationBackend for MetalBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "metal".into(),
            device: self.device.clone(),
            model: self.model_id.clone(),
            precision: self.precision.clone(),
            supports_batching: true,
        }
    }

    fn is_ready(&self) -> bool {
        self.engine.is_ready()
    }

    fn translate_batch(
        &mut self,
        inputs: &[TranslationInput],
    ) -> Result<Vec<TranslationOutput>, BackendError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let max_output_tokens = inputs
            .iter()
            .map(|input| input.max_output_tokens)
            .collect::<Vec<_>>();
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
            let pieces = self
                .source
                .encode(&input.text)
                .map_err(|error| BackendError::InvalidInput(error.to_string()))?;
            if pieces.len() + 1 > MAX_SOURCE_TOKENS {
                return Err(BackendError::InvalidInput(format!(
                    "source has {} tokens including EOS; maximum is {MAX_SOURCE_TOKENS}",
                    pieces.len() + 1
                )));
            }
            tokens.extend_from_slice(&pieces);
            tokens.push(self.eos_id);
            input_lengths.push(pieces.len() + 1);
            offsets.push(tokens.len() as u32);
        }

        let output = self
            .engine
            .translate(&tokens, &offsets, &max_output_tokens)
            .map_err(BackendError::Inference)?;
        if output.offsets.len() != inputs.len() + 1 {
            return Err(BackendError::Inference(
                "Metal engine returned malformed batch offsets".into(),
            ));
        }

        let mut results = Vec::with_capacity(inputs.len());
        for (index, &input_tokens) in input_lengths.iter().enumerate() {
            let start = output.offsets[index] as usize;
            let end = output.offsets[index + 1] as usize;
            if end < start || end > output.tokens.len() {
                return Err(BackendError::Inference(
                    "Metal engine returned out-of-range token offsets".into(),
                ));
            }
            let ids: Vec<i32> = output.tokens[start..end]
                .iter()
                .copied()
                .filter(|id| *id != self.eos_id)
                .collect();
            let text = self
                .target
                .decode(&ids)
                .map_err(|error| BackendError::Inference(error.to_string()))?;
            results.push(TranslationOutput {
                text,
                // The current greedy decoder does not expose a calibrated
                // sequence score. Do not present its placeholder as real data.
                score: None,
                input_tokens,
                output_tokens: ids.len(),
            });
        }
        Ok(results)
    }
}

/// Source-compatibility alias for callers of the pre-Metal API.
pub type MlxBackend = MetalBackend;
