use std::{io, path::PathBuf};

/// Errors produced while loading or executing Marian binary v1 Q8 models.
#[derive(Debug, thiserror::Error)]
pub enum Q8Error {
    #[error("failed to read Q8 model {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("invalid Marian binary: {0}")]
    InvalidFormat(String),

    #[error("missing tensor {0}")]
    MissingTensor(String),

    #[error("invalid tensor {name}: {reason}")]
    InvalidTensor { name: String, reason: String },

    #[error("Q8 GEMM failed: {0}")]
    Gemm(String),
}

impl Q8Error {
    pub(crate) fn tensor(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidTensor {
            name: name.into(),
            reason: reason.into(),
        }
    }
}
