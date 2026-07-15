//! Pure Rust CPU inference components for the fixed Marian Transformer-SSRU graph.
//!
//! [`CpuEngine`] executes FP32 weights and [`Q8CpuEngine`] executes Marian
//! binary v1 Q8 weights. Both implement the complete fixed graph without C++;
//! Q8 dense weights stay quantized and tied output scoring is shortlist-only.
//! Text tokenization remains separate from tensor execution.

mod backend;
mod engine;
mod legacy_q8;
mod q8_arm;
mod q8_avx2;
mod q8_engine;
mod q8_error;
mod q8_gemm;
mod segmenter;
mod tensor;

pub use backend::{CpuBackend, CpuModelBackend, Q8CpuBackend, TextTokenizer};
pub use engine::{BatchOutput, CpuEngine};
pub use legacy_q8::{
    MarianBinaryModel, MarianTensor, MarianTensorData, MarianTensorType, Q8ValidationReport,
};
pub use marian_model::{Architecture, Checksums, ModelManifest};
pub use q8_engine::Q8CpuEngine;
pub use q8_error::Q8Error;
pub use q8_gemm::{Q8ExecutionPath, Q8Linear, quantize_symmetric_u8};
pub use segmenter::{SegmentError, TextSegment, segment_text};
