use std::path::Path;

use marian_core::{
    BackendError, BackendInfo, TranslationBackend, TranslationInput, TranslationOutput,
};
use marian_cpu::segment_text;
use marian_model::ModelManifest;
use marian_tokenizer::Tokenizer;

use crate::engine::MetalEngine;

const MAX_SOURCE_TOKENS: usize = 4_096;
const MAX_TRANSLATION_SEGMENTS: usize = 4_096;

struct PlannedSegment {
    tokens: Vec<i32>,
    separator: String,
    translated: Option<String>,
    input_tokens: usize,
    output_tokens: usize,
}

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
        let precision = engine.precision().to_owned();

        Ok(Self {
            engine,
            source,
            target,
            source_lang: manifest.source_lang,
            target_lang: manifest.target_lang,
            model_id: manifest.model_id,
            precision,
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
        let mut planned = Vec::with_capacity(inputs.len());
        let mut total_segments = 0;
        for input in inputs {
            if input.source_lang != self.source_lang || input.target_lang != self.target_lang {
                return Err(BackendError::UnsupportedDirection(format!(
                    "{} -> {}; loaded model is {} -> {}",
                    input.source_lang, input.target_lang, self.source_lang, self.target_lang
                )));
            }
            let ranges = segment_text(&input.text, MAX_SOURCE_TOKENS - 1, |text| {
                self.source
                    .encode(text)
                    .map(|tokens| tokens.len())
                    .map_err(|error| error.to_string())
            })
            .map_err(|error| BackendError::InvalidInput(error.to_string()))?;
            let mut segments = Vec::with_capacity(ranges.len());
            for range in ranges {
                if total_segments >= MAX_TRANSLATION_SEGMENTS {
                    return Err(BackendError::InvalidInput(format!(
                        "batch requires more than {MAX_TRANSLATION_SEGMENTS} translated segments"
                    )));
                }
                let content = range.content(&input.text);
                let separator = range.separator(&input.text).to_owned();
                if content.is_empty() {
                    segments.push(PlannedSegment {
                        tokens: Vec::new(),
                        separator,
                        translated: Some(String::new()),
                        input_tokens: 0,
                        output_tokens: 0,
                    });
                    continue;
                }
                let mut tokens = self
                    .source
                    .encode(content)
                    .map_err(|error| BackendError::InvalidInput(error.to_string()))?;
                if tokens.len() != range.source_pieces {
                    return Err(BackendError::Inference(
                        "source tokenizer changed between segmentation and translation".into(),
                    ));
                }
                tokens.push(self.eos_id);
                let input_tokens = tokens.len();
                segments.push(PlannedSegment {
                    tokens,
                    separator,
                    translated: None,
                    input_tokens,
                    output_tokens: 0,
                });
                total_segments += 1;
            }
            planned.push(segments);
        }

        let mut positions = vec![0_usize; inputs.len()];
        let mut remaining = inputs
            .iter()
            .map(|input| input.max_output_tokens)
            .collect::<Vec<_>>();
        loop {
            for input_index in 0..planned.len() {
                while positions[input_index] < planned[input_index].len()
                    && planned[input_index][positions[input_index]]
                        .translated
                        .is_some()
                {
                    positions[input_index] += 1;
                }
                if remaining[input_index] == 0 {
                    while positions[input_index] < planned[input_index].len() {
                        planned[input_index][positions[input_index]].translated =
                            Some(String::new());
                        positions[input_index] += 1;
                    }
                }
            }
            let work = positions
                .iter()
                .enumerate()
                .filter_map(|(input_index, &segment_index)| {
                    (segment_index < planned[input_index].len())
                        .then_some((input_index, segment_index))
                })
                .collect::<Vec<_>>();
            if work.is_empty() {
                break;
            }
            let mut tokens = Vec::new();
            let mut offsets = vec![0_u32];
            let mut limits = Vec::with_capacity(work.len());
            for &(input_index, segment_index) in &work {
                tokens.extend_from_slice(&planned[input_index][segment_index].tokens);
                offsets.push(u32::try_from(tokens.len()).map_err(|_| {
                    BackendError::Inference("packed Metal input exceeds u32".into())
                })?);
                limits.push(remaining[input_index]);
            }
            let output = self
                .engine
                .translate(&tokens, &offsets, &limits)
                .map_err(BackendError::Inference)?;
            if output.offsets.len() != work.len() + 1 {
                return Err(BackendError::Inference(
                    "Metal engine returned malformed batch offsets".into(),
                ));
            }
            for (batch_index, &(input_index, segment_index)) in work.iter().enumerate() {
                let start = output.offsets[batch_index] as usize;
                let end = output.offsets[batch_index + 1] as usize;
                if end < start || end > output.tokens.len() {
                    return Err(BackendError::Inference(
                        "Metal engine returned out-of-range token offsets".into(),
                    ));
                }
                let ids = output.tokens[start..end]
                    .iter()
                    .copied()
                    .filter(|id| *id != self.eos_id)
                    .collect::<Vec<_>>();
                let text = self
                    .target
                    .decode(&ids)
                    .map_err(|error| BackendError::Inference(error.to_string()))?;
                planned[input_index][segment_index].translated = Some(text);
                planned[input_index][segment_index].output_tokens = ids.len();
                remaining[input_index] = remaining[input_index].saturating_sub(ids.len());
                positions[input_index] += 1;
            }
        }

        planned
            .into_iter()
            .map(|segments| {
                let mut text = String::new();
                let mut input_tokens = 0;
                let mut output_tokens = 0;
                for segment in segments {
                    text.push_str(segment.translated.as_deref().ok_or_else(|| {
                        BackendError::Inference("Metal segment was not translated".into())
                    })?);
                    text.push_str(&segment.separator);
                    input_tokens += segment.input_tokens;
                    output_tokens += segment.output_tokens;
                }
                Ok(TranslationOutput {
                    text,
                    score: None,
                    input_tokens,
                    output_tokens,
                })
            })
            .collect()
    }
}

/// Source-compatibility alias for callers of the pre-Metal API.
pub type MlxBackend = MetalBackend;
