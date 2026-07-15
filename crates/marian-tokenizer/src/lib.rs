use std::path::{Path, PathBuf};

use sentencepiece_rust::SentencePieceProcessor;
use thiserror::Error;

/// A narrow, pure-Rust SentencePiece inference wrapper used by Marian backends.
pub struct Tokenizer {
    processor: SentencePieceProcessor,
}

impl Tokenizer {
    /// Opens a serialized SentencePiece model without loading a native library.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        let processor =
            SentencePieceProcessor::open(path).map_err(|source| TokenizerError::Open {
                path: path.to_path_buf(),
                source,
            })?;
        Ok(Self { processor })
    }

    /// Returns the number of pieces in the model vocabulary.
    pub fn len(&self) -> usize {
        self.processor.piece_size()
    }

    /// Returns whether the model vocabulary contains no pieces.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Encodes UTF-8 text into SentencePiece vocabulary ids.
    pub fn encode(&self, text: &str) -> Result<Vec<i32>, TokenizerError> {
        self.processor.encode(text).map_err(TokenizerError::Encode)
    }

    /// Decodes SentencePiece vocabulary ids into normalized UTF-8 text.
    pub fn decode(&self, ids: &[i32]) -> Result<String, TokenizerError> {
        self.processor.decode(ids).map_err(TokenizerError::Decode)
    }
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("failed to open SentencePiece model {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: sentencepiece_rust::Error,
    },
    #[error("SentencePiece encoding failed: {0}")]
    Encode(#[source] sentencepiece_rust::Error),
    #[error("SentencePiece decoding failed: {0}")]
    Decode(#[source] sentencepiece_rust::Error),
}
