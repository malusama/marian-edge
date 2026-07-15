use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslationInput {
    pub text: String,
    pub source_lang: String,
    pub target_lang: String,
    pub max_output_tokens: usize,
}

impl TranslationInput {
    pub fn new(
        text: impl Into<String>,
        source_lang: impl Into<String>,
        target_lang: impl Into<String>,
    ) -> Self {
        Self {
            text: text.into(),
            source_lang: source_lang.into(),
            target_lang: target_lang.into(),
            max_output_tokens: 512,
        }
    }

    pub(crate) fn batch_key(&self) -> (&str, &str, usize, usize) {
        let characters = self.text.chars().count().max(1);
        let length_bucket = characters.checked_next_power_of_two().unwrap_or(usize::MAX);
        (
            &self.source_lang,
            &self.target_lang,
            self.max_output_tokens,
            length_bucket,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationOutput {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackendInfo {
    pub name: String,
    pub device: String,
    pub model: String,
    pub precision: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention: Option<String>,
    pub supports_batching: bool,
}

#[derive(Debug, Error, Clone)]
pub enum BackendError {
    #[error("invalid request: {0}")]
    InvalidInput(String),
    #[error("unsupported language direction: {0}")]
    UnsupportedDirection(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("inference error: {0}")]
    Inference(String),
}

#[derive(Debug, Error, Clone)]
pub enum TranslateError {
    #[error("translation queue is full")]
    QueueFull,
    #[error("translation service is shutting down")]
    ShuttingDown,
    #[error("translation timed out after {0:?}")]
    Timeout(Duration),
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error("inference worker stopped unexpectedly")]
    WorkerStopped,
}
