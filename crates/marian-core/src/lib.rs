//! Runtime-independent translation types and a bounded dynamic-batch scheduler.
//!
//! The scheduler deliberately owns a backend on one dedicated OS thread. That
//! gives direct Metal a single command-queue owner while Tokio remains free to
//! serve many concurrent HTTP requests.

mod backend;
#[cfg(not(target_arch = "wasm32"))]
mod scheduler;
mod segmenter;
mod types;

pub use backend::{EchoBackend, TranslationBackend};
#[cfg(not(target_arch = "wasm32"))]
pub use scheduler::{SchedulerConfig, SchedulerStats, StatsSnapshot, Translator};
pub use segmenter::{
    MAX_COUNTED_PIECES, MAX_ENCODING_BYTES, MAX_ENCODING_CALLS, MAX_SEGMENTER_INPUT_BYTES,
    MAX_SEGMENTS, SegmentError, TextSegment, segment_text,
};
pub use types::{BackendError, BackendInfo, TranslateError, TranslationInput, TranslationOutput};
