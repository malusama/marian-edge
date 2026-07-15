//! CPU execution limits shared by the FP32/Q8 engines and text adapter.

pub(crate) const MAXIMUM_SOURCE_TOKENS: usize = 256;
pub(crate) const MAXIMUM_GENERATION_STEPS: usize = 256;
pub(crate) const MAXIMUM_ENGINE_BATCH: usize = 256;
pub(crate) const MAXIMUM_PADDED_ATTENTION_CELLS: usize = 65_536;
pub(crate) const MAXIMUM_TRANSLATION_SEGMENTS: usize = 4_096;
