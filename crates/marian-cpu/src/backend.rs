use std::path::Path;

use marian_core::{
    BackendError, BackendInfo, TranslationBackend, TranslationInput, TranslationOutput,
};
use marian_model::ModelManifest;

use crate::{
    engine::{BatchOutput, CpuEngine},
    limits::{
        MAXIMUM_ENGINE_BATCH, MAXIMUM_PADDED_ATTENTION_CELLS, MAXIMUM_SOURCE_TOKENS,
        MAXIMUM_TRANSLATION_SEGMENTS,
    },
    q8_engine::Q8CpuEngine,
    segment_text,
};

/// Minimal text/token boundary needed by [`CpuBackend`].
///
/// A pure Rust SentencePiece crate can implement this directly, or callers can
/// use a small local newtype adapter. The tensor executor itself never depends
/// on a tokenizer implementation.
pub trait TextTokenizer: 'static {
    fn vocabulary_size(&self) -> usize;

    fn encode(&self, text: &str) -> Result<Vec<i32>, String>;

    fn decode(&self, token_ids: &[i32]) -> Result<String, String>;
}

impl TextTokenizer for marian_tokenizer::Tokenizer {
    fn vocabulary_size(&self) -> usize {
        self.len()
    }

    fn encode(&self, text: &str) -> Result<Vec<i32>, String> {
        self.encode(text).map_err(|error| error.to_string())
    }

    fn decode(&self, token_ids: &[i32]) -> Result<String, String> {
        self.decode(token_ids).map_err(|error| error.to_string())
    }
}

trait CpuTokenEngine {
    fn translate_token_ids(
        &self,
        tokens: &[i32],
        offsets: &[u32],
        max_output_tokens: &[usize],
    ) -> Result<BatchOutput, String>;

    fn label(&self) -> &'static str;
}

impl CpuTokenEngine for CpuEngine {
    fn translate_token_ids(
        &self,
        tokens: &[i32],
        offsets: &[u32],
        max_output_tokens: &[usize],
    ) -> Result<BatchOutput, String> {
        CpuEngine::translate_token_ids(self, tokens, offsets, max_output_tokens)
    }

    fn label(&self) -> &'static str {
        "CPU"
    }
}

impl CpuTokenEngine for Q8CpuEngine {
    fn translate_token_ids(
        &self,
        tokens: &[i32],
        offsets: &[u32],
        max_output_tokens: &[usize],
    ) -> Result<BatchOutput, String> {
        Q8CpuEngine::translate_token_ids(self, tokens, offsets, max_output_tokens)
    }

    fn label(&self) -> &'static str {
        "Q8 CPU"
    }
}

struct PlannedInput {
    segments: Vec<PlannedOutput>,
}

struct PlannedOutput {
    translated: Option<String>,
    separator: String,
    input_tokens: usize,
    output_tokens: usize,
}

struct SegmentWork {
    input_index: usize,
    segment_index: usize,
    tokens: Vec<i32>,
}

#[allow(clippy::too_many_arguments)]
fn translate_with_engine(
    engine: &dyn CpuTokenEngine,
    source: &dyn TextTokenizer,
    target: &dyn TextTokenizer,
    source_lang: &str,
    target_lang: &str,
    eos_id: i32,
    inputs: &[TranslationInput],
) -> Result<Vec<TranslationOutput>, BackendError> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    for input in inputs {
        if input.source_lang != source_lang || input.target_lang != target_lang {
            return Err(BackendError::UnsupportedDirection(format!(
                "{} -> {}; loaded model is {source_lang} -> {target_lang}",
                input.source_lang, input.target_lang
            )));
        }
    }

    let mut planned_inputs = Vec::with_capacity(inputs.len());
    let mut work = (0..inputs.len()).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut total_work = 0_usize;
    for (input_index, input) in inputs.iter().enumerate() {
        let segments = segment_text(&input.text, MAXIMUM_SOURCE_TOKENS - 1, |text| {
            source.encode(text).map(|pieces| pieces.len())
        })
        .map_err(|error| BackendError::InvalidInput(error.to_string()))?;
        let mut planned = PlannedInput {
            segments: Vec::with_capacity(segments.len()),
        };
        for segment in segments {
            let content = segment.content(&input.text);
            let segment_index = planned.segments.len();
            let separator = segment.separator(&input.text).to_owned();
            if content.is_empty() {
                planned.segments.push(PlannedOutput {
                    translated: Some(String::new()),
                    separator,
                    input_tokens: 0,
                    output_tokens: 0,
                });
                continue;
            }

            if total_work >= MAXIMUM_TRANSLATION_SEGMENTS {
                return Err(BackendError::InvalidInput(format!(
                    "batch requires more than {MAXIMUM_TRANSLATION_SEGMENTS} translated segments"
                )));
            }
            let mut pieces = source.encode(content).map_err(|error| {
                BackendError::InvalidInput(format!("tokenizer encoding failed: {error}"))
            })?;
            if pieces.len() != segment.source_pieces {
                return Err(BackendError::Inference(format!(
                    "source tokenizer returned {} pieces after sizing the segment as {}",
                    pieces.len(),
                    segment.source_pieces
                )));
            }
            if let Some(&invalid) = pieces
                .iter()
                .find(|&&token| token < 0 || token as usize >= source.vocabulary_size())
            {
                return Err(BackendError::InvalidInput(format!(
                    "tokenizer returned source token {invalid} outside vocabulary {}",
                    source.vocabulary_size()
                )));
            }
            pieces.push(eos_id);
            if pieces.len() > MAXIMUM_SOURCE_TOKENS {
                return Err(BackendError::Inference(format!(
                    "segmented source has {} tokens including EOS; maximum is {MAXIMUM_SOURCE_TOKENS}",
                    pieces.len()
                )));
            }
            let input_tokens = pieces.len();
            planned.segments.push(PlannedOutput {
                translated: None,
                separator,
                input_tokens,
                output_tokens: 0,
            });
            work[input_index].push(SegmentWork {
                input_index,
                segment_index,
                tokens: pieces,
            });
            total_work += 1;
        }
        planned_inputs.push(planned);
    }

    let mut positions = vec![0_usize; work.len()];
    let mut remaining_output_tokens = inputs
        .iter()
        .map(|input| input.max_output_tokens)
        .collect::<Vec<_>>();
    loop {
        for input_index in 0..work.len() {
            if remaining_output_tokens[input_index] != 0 {
                continue;
            }
            while positions[input_index] < work[input_index].len() {
                let segment = &work[input_index][positions[input_index]];
                let destination = &mut planned_inputs[input_index].segments[segment.segment_index];
                if destination.translated.replace(String::new()).is_some() {
                    return Err(BackendError::Inference(
                        "CPU segment was translated more than once".into(),
                    ));
                }
                positions[input_index] += 1;
            }
        }

        let mut batch = Vec::new();
        let mut maximum_length = 0;
        for (input_index, segments) in work.iter().enumerate() {
            let Some(segment) = segments.get(positions[input_index]) else {
                continue;
            };
            if batch.len() >= MAXIMUM_ENGINE_BATCH {
                break;
            }
            let candidate_length = maximum_length.max(segment.tokens.len());
            let candidate_batch = batch.len() + 1;
            let attention_cells = candidate_batch
                .checked_mul(candidate_length)
                .and_then(|value| value.checked_mul(candidate_length))
                .unwrap_or(usize::MAX);
            if attention_cells > MAXIMUM_PADDED_ATTENTION_CELLS {
                continue;
            }
            maximum_length = candidate_length;
            batch.push(segment);
        }
        if batch.is_empty() {
            if positions
                .iter()
                .zip(&work)
                .all(|(&position, segments)| position == segments.len())
            {
                break;
            }
            return Err(BackendError::Inference(
                "unable to construct a bounded CPU inference batch".into(),
            ));
        }

        let mut tokens = Vec::new();
        let mut offsets = Vec::with_capacity(batch.len() + 1);
        let mut output_limits = Vec::with_capacity(batch.len());
        offsets.push(0_u32);
        for segment in &batch {
            tokens.extend_from_slice(&segment.tokens);
            offsets.push(u32::try_from(tokens.len()).map_err(|_| {
                BackendError::InvalidInput("packed source token count exceeds u32".into())
            })?);
            output_limits.push(remaining_output_tokens[segment.input_index]);
        }

        let output = engine
            .translate_token_ids(&tokens, &offsets, &output_limits)
            .map_err(BackendError::Inference)?;
        if output.offsets.len() != batch.len() + 1 {
            return Err(BackendError::Inference(format!(
                "{} engine returned malformed batch offsets",
                engine.label()
            )));
        }
        for (row, segment) in batch.iter().enumerate() {
            let start = output.offsets[row] as usize;
            let end = output.offsets[row + 1] as usize;
            if end < start || end > output.tokens.len() {
                return Err(BackendError::Inference(format!(
                    "{} engine returned out-of-range token offsets",
                    engine.label()
                )));
            }
            let generated_tokens = end - start;
            remaining_output_tokens[segment.input_index] = remaining_output_tokens
                [segment.input_index]
                .checked_sub(generated_tokens)
                .ok_or_else(|| {
                    BackendError::Inference(format!(
                        "{} engine exceeded the original max_output_tokens budget",
                        engine.label()
                    ))
                })?;
            let ids = output.tokens[start..end]
                .iter()
                .copied()
                .filter(|id| *id != eos_id)
                .collect::<Vec<_>>();
            let text = target.decode(&ids).map_err(|error| {
                BackendError::Inference(format!("tokenizer decoding failed: {error}"))
            })?;
            let destination =
                &mut planned_inputs[segment.input_index].segments[segment.segment_index];
            if destination.translated.replace(text).is_some() {
                return Err(BackendError::Inference(
                    "CPU segment was translated more than once".into(),
                ));
            }
            destination.output_tokens = ids.len();
            positions[segment.input_index] += 1;
        }
    }

    planned_inputs
        .into_iter()
        .map(|planned| {
            let mut text = String::new();
            let mut input_tokens = 0_usize;
            let mut output_tokens = 0_usize;
            for segment in planned.segments {
                let translated = segment.translated.ok_or_else(|| {
                    BackendError::Inference("CPU segment has no translation result".into())
                })?;
                text.push_str(&translated);
                text.push_str(&segment.separator);
                input_tokens = input_tokens
                    .checked_add(segment.input_tokens)
                    .ok_or_else(|| {
                        BackendError::Inference("input token count overflowed usize".into())
                    })?;
                output_tokens = output_tokens
                    .checked_add(segment.output_tokens)
                    .ok_or_else(|| {
                        BackendError::Inference("output token count overflowed usize".into())
                    })?;
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

/// `marian_core` backend backed by the pure Rust FP32 executor.
pub struct CpuBackend {
    engine: CpuEngine,
    source: Box<dyn TextTokenizer>,
    target: Box<dyn TextTokenizer>,
    source_lang: String,
    target_lang: String,
    model_id: String,
    precision: String,
    eos_id: i32,
}

impl CpuBackend {
    /// Loads the model and its pure Rust SentencePiece tokenizers.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, BackendError> {
        let model_dir = std::fs::canonicalize(model_dir.as_ref()).map_err(|error| {
            BackendError::Model(format!(
                "failed to resolve model directory {}: {error}",
                model_dir.as_ref().display()
            ))
        })?;
        let manifest = ModelManifest::load(&model_dir)?;
        manifest.verify_runtime_files(&model_dir)?;
        let source_path = model_dir.join(&manifest.source_vocab);
        let target_path = model_dir.join(&manifest.target_vocab);
        let source = marian_tokenizer::Tokenizer::open(&source_path).map_err(|error| {
            BackendError::Model(format!(
                "failed to load source tokenizer {}: {error}",
                source_path.display()
            ))
        })?;
        let target = marian_tokenizer::Tokenizer::open(&target_path).map_err(|error| {
            BackendError::Model(format!(
                "failed to load target tokenizer {}: {error}",
                target_path.display()
            ))
        })?;
        Self::load_verified(model_dir, manifest, Box::new(source), Box::new(target))
    }

    pub fn load_with_tokenizers<S, T>(
        model_dir: impl AsRef<Path>,
        source: S,
        target: T,
    ) -> Result<Self, BackendError>
    where
        S: TextTokenizer,
        T: TextTokenizer,
    {
        Self::load_with_boxed_tokenizers(model_dir, Box::new(source), Box::new(target))
    }

    pub fn load_with_boxed_tokenizers(
        model_dir: impl AsRef<Path>,
        source: Box<dyn TextTokenizer>,
        target: Box<dyn TextTokenizer>,
    ) -> Result<Self, BackendError> {
        let model_dir = std::fs::canonicalize(model_dir.as_ref()).map_err(|error| {
            BackendError::Model(format!(
                "failed to resolve model directory {}: {error}",
                model_dir.as_ref().display()
            ))
        })?;
        let manifest = ModelManifest::load(&model_dir)?;
        manifest.verify_runtime_files(&model_dir)?;
        Self::load_verified(model_dir, manifest, source, target)
    }

    fn load_verified(
        model_dir: impl AsRef<Path>,
        manifest: ModelManifest,
        source: Box<dyn TextTokenizer>,
        target: Box<dyn TextTokenizer>,
    ) -> Result<Self, BackendError> {
        let model_dir = model_dir.as_ref();
        if manifest.precision != "fp32" {
            return Err(BackendError::Model(format!(
                "FP32 CPU backend requires precision fp32, got {}",
                manifest.precision
            )));
        }
        if source.vocabulary_size() != manifest.architecture.source_vocab_size
            || target.vocabulary_size() != manifest.architecture.target_vocab_size
        {
            return Err(BackendError::Model(format!(
                "tokenizer vocabulary sizes {} / {} do not match model {} / {}",
                source.vocabulary_size(),
                target.vocabulary_size(),
                manifest.architecture.source_vocab_size,
                manifest.architecture.target_vocab_size
            )));
        }

        let weights = model_dir.join(&manifest.weights);
        let shortlist = manifest.shortlist.as_ref().map(|path| model_dir.join(path));
        let engine = CpuEngine::load(&weights, shortlist.as_deref(), &manifest.architecture)
            .map_err(BackendError::Model)?;
        engine
            .warmup()
            .map_err(|error| BackendError::Inference(format!("CPU warmup failed: {error}")))?;

        Ok(Self {
            engine,
            source,
            target,
            source_lang: manifest.source_lang,
            target_lang: manifest.target_lang,
            model_id: manifest.model_id,
            precision: manifest.precision,
            eos_id: manifest.architecture.eos_id,
        })
    }

    pub fn engine(&self) -> &CpuEngine {
        &self.engine
    }
}

impl TranslationBackend for CpuBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "cpu".into(),
            device: std::env::consts::ARCH.into(),
            model: self.model_id.clone(),
            precision: self.precision.clone(),
            attention: Some("streaming-exact-simd-value".into()),
            supports_batching: true,
        }
    }

    fn translate_batch(
        &mut self,
        inputs: &[TranslationInput],
    ) -> Result<Vec<TranslationOutput>, BackendError> {
        translate_with_engine(
            &self.engine,
            self.source.as_ref(),
            self.target.as_ref(),
            &self.source_lang,
            &self.target_lang,
            self.eos_id,
            inputs,
        )
    }
}

/// `marian_core` backend backed by the pure Rust Q8 executor.
pub struct Q8CpuBackend {
    engine: Q8CpuEngine,
    source: Box<dyn TextTokenizer>,
    target: Box<dyn TextTokenizer>,
    source_lang: String,
    target_lang: String,
    model_id: String,
    eos_id: i32,
}

impl Q8CpuBackend {
    /// Load a checksum-verified `precision: "q8"` model directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, BackendError> {
        let model_dir = std::fs::canonicalize(model_dir.as_ref()).map_err(|error| {
            BackendError::Model(format!(
                "failed to resolve model directory {}: {error}",
                model_dir.as_ref().display()
            ))
        })?;
        let manifest = ModelManifest::load(&model_dir)?;
        manifest.verify_runtime_files(&model_dir)?;
        let source_path = model_dir.join(&manifest.source_vocab);
        let target_path = model_dir.join(&manifest.target_vocab);
        let source = marian_tokenizer::Tokenizer::open(&source_path).map_err(|error| {
            BackendError::Model(format!(
                "failed to load source tokenizer {}: {error}",
                source_path.display()
            ))
        })?;
        let target = marian_tokenizer::Tokenizer::open(&target_path).map_err(|error| {
            BackendError::Model(format!(
                "failed to load target tokenizer {}: {error}",
                target_path.display()
            ))
        })?;
        Self::load_verified(model_dir, manifest, Box::new(source), Box::new(target))
    }

    /// Loads a complete Q8 backend from byte payloads, for sandboxed runtimes
    /// such as Cloudflare Workers where filesystem access is unavailable.
    pub fn from_bytes(
        manifest_bytes: &[u8],
        weights_bytes: &[u8],
        source_vocab_bytes: &[u8],
        target_vocab_bytes: &[u8],
        shortlist_bytes: Option<&[u8]>,
    ) -> Result<Self, BackendError> {
        let manifest = ModelManifest::from_bytes(manifest_bytes)?;
        manifest.verify_runtime_bytes(
            weights_bytes,
            source_vocab_bytes,
            target_vocab_bytes,
            shortlist_bytes,
        )?;
        let source =
            marian_tokenizer::Tokenizer::from_bytes(source_vocab_bytes).map_err(|error| {
                BackendError::Model(format!("failed to load source tokenizer: {error}"))
            })?;
        let target =
            marian_tokenizer::Tokenizer::from_bytes(target_vocab_bytes).map_err(|error| {
                BackendError::Model(format!("failed to load target tokenizer: {error}"))
            })?;
        Self::from_verified_bytes(
            manifest,
            weights_bytes,
            shortlist_bytes,
            Box::new(source),
            Box::new(target),
        )
    }

    /// Owned-buffer variant used by Wasm hosts so the raw model allocation can
    /// be released before architecture-specific packing starts.
    pub fn from_owned_bytes(
        manifest_bytes: Vec<u8>,
        weights_bytes: Vec<u8>,
        source_vocab_bytes: Vec<u8>,
        target_vocab_bytes: Vec<u8>,
        shortlist_bytes: Option<Vec<u8>>,
    ) -> Result<Self, BackendError> {
        let manifest = ModelManifest::from_bytes(&manifest_bytes)?;
        drop(manifest_bytes);
        manifest.verify_runtime_bytes(
            &weights_bytes,
            &source_vocab_bytes,
            &target_vocab_bytes,
            shortlist_bytes.as_deref(),
        )?;
        Self::from_preverified_parts(
            manifest,
            weights_bytes,
            source_vocab_bytes,
            target_vocab_bytes,
            shortlist_bytes,
        )
    }

    /// Loads caller-verified payloads. The host must authenticate every byte
    /// against the manifest before using this entry point.
    pub fn from_preverified_owned_bytes(
        manifest_bytes: Vec<u8>,
        weights_bytes: Vec<u8>,
        source_vocab_bytes: Vec<u8>,
        target_vocab_bytes: Vec<u8>,
        shortlist_bytes: Option<Vec<u8>>,
    ) -> Result<Self, BackendError> {
        let manifest = ModelManifest::from_bytes(&manifest_bytes)?;
        drop(manifest_bytes);
        Self::from_preverified_parts(
            manifest,
            weights_bytes,
            source_vocab_bytes,
            target_vocab_bytes,
            shortlist_bytes,
        )
    }

    fn from_preverified_parts(
        manifest: ModelManifest,
        weights_bytes: Vec<u8>,
        source_vocab_bytes: Vec<u8>,
        target_vocab_bytes: Vec<u8>,
        shortlist_bytes: Option<Vec<u8>>,
    ) -> Result<Self, BackendError> {
        let source =
            marian_tokenizer::Tokenizer::from_bytes(&source_vocab_bytes).map_err(|error| {
                BackendError::Model(format!("failed to load source tokenizer: {error}"))
            })?;
        drop(source_vocab_bytes);
        let target =
            marian_tokenizer::Tokenizer::from_bytes(&target_vocab_bytes).map_err(|error| {
                BackendError::Model(format!("failed to load target tokenizer: {error}"))
            })?;
        drop(target_vocab_bytes);
        Self::from_verified_owned_bytes(
            manifest,
            weights_bytes,
            shortlist_bytes,
            Box::new(source),
            Box::new(target),
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_with_tokenizers<S, T>(
        model_dir: impl AsRef<Path>,
        source: S,
        target: T,
    ) -> Result<Self, BackendError>
    where
        S: TextTokenizer,
        T: TextTokenizer,
    {
        let model_dir = std::fs::canonicalize(model_dir.as_ref()).map_err(|error| {
            BackendError::Model(format!(
                "failed to resolve model directory {}: {error}",
                model_dir.as_ref().display()
            ))
        })?;
        let manifest = ModelManifest::load(&model_dir)?;
        manifest.verify_runtime_files(&model_dir)?;
        Self::load_verified(model_dir, manifest, Box::new(source), Box::new(target))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_verified(
        model_dir: impl AsRef<Path>,
        manifest: ModelManifest,
        source: Box<dyn TextTokenizer>,
        target: Box<dyn TextTokenizer>,
    ) -> Result<Self, BackendError> {
        if manifest.precision != "q8" {
            return Err(BackendError::Model(format!(
                "Q8 CPU backend requires precision q8, got {}",
                manifest.precision
            )));
        }
        if source.vocabulary_size() != manifest.architecture.source_vocab_size
            || target.vocabulary_size() != manifest.architecture.target_vocab_size
        {
            return Err(BackendError::Model(format!(
                "tokenizer vocabulary sizes {} / {} do not match model {} / {}",
                source.vocabulary_size(),
                target.vocabulary_size(),
                manifest.architecture.source_vocab_size,
                manifest.architecture.target_vocab_size
            )));
        }

        let model_dir = model_dir.as_ref();
        let weights = model_dir.join(&manifest.weights);
        let shortlist = manifest.shortlist.as_ref().map(|path| model_dir.join(path));
        let engine = Q8CpuEngine::load(&weights, shortlist.as_deref(), &manifest.architecture)
            .map_err(BackendError::Model)?;
        #[cfg(not(target_arch = "wasm32"))]
        engine
            .warmup()
            .map_err(|error| BackendError::Inference(format!("Q8 CPU warmup failed: {error}")))?;

        Ok(Self {
            engine,
            source,
            target,
            source_lang: manifest.source_lang,
            target_lang: manifest.target_lang,
            model_id: manifest.model_id,
            eos_id: manifest.architecture.eos_id,
        })
    }

    fn from_verified_bytes(
        manifest: ModelManifest,
        weights: &[u8],
        shortlist: Option<&[u8]>,
        source: Box<dyn TextTokenizer>,
        target: Box<dyn TextTokenizer>,
    ) -> Result<Self, BackendError> {
        if manifest.precision != "q8" {
            return Err(BackendError::Model(format!(
                "Q8 CPU backend requires precision q8, got {}",
                manifest.precision
            )));
        }
        if source.vocabulary_size() != manifest.architecture.source_vocab_size
            || target.vocabulary_size() != manifest.architecture.target_vocab_size
        {
            return Err(BackendError::Model(format!(
                "tokenizer vocabulary sizes {} / {} do not match model {} / {}",
                source.vocabulary_size(),
                target.vocabulary_size(),
                manifest.architecture.source_vocab_size,
                manifest.architecture.target_vocab_size
            )));
        }
        let engine = Q8CpuEngine::from_bytes(weights, shortlist, &manifest.architecture)
            .map_err(BackendError::Model)?;
        engine
            .warmup()
            .map_err(|error| BackendError::Inference(format!("Q8 CPU warmup failed: {error}")))?;
        Ok(Self {
            engine,
            source,
            target,
            source_lang: manifest.source_lang,
            target_lang: manifest.target_lang,
            model_id: manifest.model_id,
            eos_id: manifest.architecture.eos_id,
        })
    }

    fn from_verified_owned_bytes(
        manifest: ModelManifest,
        weights: Vec<u8>,
        shortlist: Option<Vec<u8>>,
        source: Box<dyn TextTokenizer>,
        target: Box<dyn TextTokenizer>,
    ) -> Result<Self, BackendError> {
        if manifest.precision != "q8" {
            return Err(BackendError::Model(format!(
                "Q8 CPU backend requires precision q8, got {}",
                manifest.precision
            )));
        }
        if source.vocabulary_size() != manifest.architecture.source_vocab_size
            || target.vocabulary_size() != manifest.architecture.target_vocab_size
        {
            return Err(BackendError::Model(format!(
                "tokenizer vocabulary sizes {} / {} do not match model {} / {}",
                source.vocabulary_size(),
                target.vocabulary_size(),
                manifest.architecture.source_vocab_size,
                manifest.architecture.target_vocab_size
            )));
        }
        let engine =
            Q8CpuEngine::from_owned_bytes(weights, shortlist.as_deref(), &manifest.architecture)
                .map_err(BackendError::Model)?;
        drop(shortlist);
        #[cfg(not(target_arch = "wasm32"))]
        engine
            .warmup()
            .map_err(|error| BackendError::Inference(format!("Q8 CPU warmup failed: {error}")))?;
        Ok(Self {
            engine,
            source,
            target,
            source_lang: manifest.source_lang,
            target_lang: manifest.target_lang,
            model_id: manifest.model_id,
            eos_id: manifest.architecture.eos_id,
        })
    }

    pub fn engine(&self) -> &Q8CpuEngine {
        &self.engine
    }
}

impl TranslationBackend for Q8CpuBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "cpu".into(),
            device: std::env::consts::ARCH.into(),
            model: self.model_id.clone(),
            precision: "q8".into(),
            attention: Some("streaming-exact-simd-value".into()),
            supports_batching: true,
        }
    }

    fn translate_batch(
        &mut self,
        inputs: &[TranslationInput],
    ) -> Result<Vec<TranslationOutput>, BackendError> {
        translate_with_engine(
            &self.engine,
            self.source.as_ref(),
            self.target.as_ref(),
            &self.source_lang,
            &self.target_lang,
            self.eos_id,
            inputs,
        )
    }
}

/// Manifest-dispatched pure Rust CPU backend.
///
/// This is the production entry point: `precision: "fp32"` selects
/// [`CpuBackend`] and `precision: "q8"` selects [`Q8CpuBackend`]. Keeping the
/// concrete backends public preserves direct engine access for differential
/// and artifact tests without making the HTTP service duplicate dispatch.
pub enum CpuModelBackend {
    Fp32(CpuBackend),
    Q8(Q8CpuBackend),
}

impl CpuModelBackend {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, BackendError> {
        let manifest = ModelManifest::load(model_dir.as_ref())?;
        match manifest.precision.as_str() {
            "fp32" => CpuBackend::load(model_dir).map(Self::Fp32),
            "q8" => Q8CpuBackend::load(model_dir).map(Self::Q8),
            precision => Err(BackendError::Model(format!(
                "unsupported pure Rust CPU precision {precision}"
            ))),
        }
    }
}

impl TranslationBackend for CpuModelBackend {
    fn info(&self) -> BackendInfo {
        match self {
            Self::Fp32(backend) => backend.info(),
            Self::Q8(backend) => backend.info(),
        }
    }

    fn is_ready(&self) -> bool {
        match self {
            Self::Fp32(backend) => backend.is_ready(),
            Self::Q8(backend) => backend.is_ready(),
        }
    }

    fn translate_batch(
        &mut self,
        inputs: &[TranslationInput],
    ) -> Result<Vec<TranslationOutput>, BackendError> {
        match self {
            Self::Fp32(backend) => backend.translate_batch(inputs),
            Self::Q8(backend) => backend.translate_batch(inputs),
        }
    }
}

#[cfg(test)]
mod tests {
    use marian_core::TranslationInput;

    use super::{BatchOutput, CpuTokenEngine, TextTokenizer, translate_with_engine};

    struct WordTokenizer;

    impl TextTokenizer for WordTokenizer {
        fn vocabulary_size(&self) -> usize {
            32
        }

        fn encode(&self, text: &str) -> Result<Vec<i32>, String> {
            Ok(text.split_whitespace().map(|_| 2).collect())
        }

        fn decode(&self, token_ids: &[i32]) -> Result<String, String> {
            Ok("x".repeat(token_ids.len()))
        }
    }

    struct LimitEngine;

    impl CpuTokenEngine for LimitEngine {
        fn translate_token_ids(
            &self,
            _tokens: &[i32],
            offsets: &[u32],
            max_output_tokens: &[usize],
        ) -> Result<BatchOutput, String> {
            let mut output = BatchOutput {
                tokens: Vec::new(),
                offsets: vec![0],
            };
            assert_eq!(offsets.len(), max_output_tokens.len() + 1);
            for &limit in max_output_tokens {
                output.tokens.extend(std::iter::repeat_n(7, limit.min(2)));
                output.offsets.push(output.tokens.len() as u32);
            }
            Ok(output)
        }

        fn label(&self) -> &'static str {
            "test"
        }
    }

    #[test]
    fn segmented_text_shares_one_original_output_budget() {
        let mut first = TranslationInput::new("One. Two.", "en", "zh");
        first.max_output_tokens = 1;
        let mut second = TranslationInput::new("Three. Four.", "en", "zh");
        second.max_output_tokens = 1;

        let outputs = translate_with_engine(
            &LimitEngine,
            &WordTokenizer,
            &WordTokenizer,
            "en",
            "zh",
            0,
            &[first, second],
        )
        .unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].text, "x ");
        assert_eq!(outputs[1].text, "x ");
        assert_eq!(outputs[0].output_tokens, 1);
        assert_eq!(outputs[1].output_tokens, 1);
    }
}
