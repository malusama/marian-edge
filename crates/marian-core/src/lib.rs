//! Runtime-independent translation types and a bounded dynamic-batch scheduler.
//!
//! The scheduler deliberately owns a backend on one dedicated OS thread. That
//! gives direct Metal a single command-queue owner while Tokio remains free to
//! serve many concurrent HTTP requests.

mod backend;
mod scheduler;
mod types;

pub use backend::{EchoBackend, TranslationBackend};
pub use scheduler::{SchedulerConfig, SchedulerStats, StatsSnapshot, Translator};
pub use types::{BackendError, BackendInfo, TranslateError, TranslationInput, TranslationOutput};
