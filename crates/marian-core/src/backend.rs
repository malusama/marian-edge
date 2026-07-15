use crate::{BackendError, BackendInfo, TranslationInput, TranslationOutput};

/// A synchronous inference backend owned exclusively by the scheduler thread.
///
/// It intentionally does not require `Send` or `Sync`: accelerator objects and
/// CPU model state are created, used, and destroyed on one dedicated thread.
pub trait TranslationBackend: 'static {
    fn info(&self) -> BackendInfo;

    /// Whether the backend can accept another inference request.
    ///
    /// Backends backed by an external worker should return `false` after a
    /// terminal transport failure. The scheduler will then reject new work
    /// and drain already-admitted requests instead of continuing to enqueue
    /// against a dead worker.
    fn is_ready(&self) -> bool {
        true
    }

    /// Minimum number of identical rows to retain when the scheduler
    /// coalesces duplicates inside one dynamic batch. Accelerators may prefer
    /// a wider matrix even though every retained row produces the same result.
    fn preferred_duplicate_batch_width(&self) -> usize {
        1
    }

    fn translate_batch(
        &mut self,
        inputs: &[TranslationInput],
    ) -> Result<Vec<TranslationOutput>, BackendError>;
}

/// Test/development backend. It is never selected implicitly in production.
pub struct EchoBackend;

impl TranslationBackend for EchoBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "echo".into(),
            device: "none".into(),
            model: "development-only".into(),
            precision: "n/a".into(),
            attention: None,
            supports_batching: true,
        }
    }

    fn translate_batch(
        &mut self,
        inputs: &[TranslationInput],
    ) -> Result<Vec<TranslationOutput>, BackendError> {
        Ok(inputs
            .iter()
            .map(|input| TranslationOutput {
                text: input.text.clone(),
                score: None,
                input_tokens: input.text.split_whitespace().count(),
                output_tokens: input.text.split_whitespace().count(),
            })
            .collect())
    }
}
