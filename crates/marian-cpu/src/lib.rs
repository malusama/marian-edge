//! Pure Rust CPU inference components for the fixed Marian Transformer-SSRU graph.
//!
//! [`CpuEngine`] executes FP32 weights and [`Q8CpuEngine`] executes Marian
//! binary v1 Q8 weights. Both implement the complete fixed graph without C++;
//! Q8 dense weights stay quantized and tied output scoring is shortlist-only.
//! Text tokenization remains separate from tensor execution.

mod backend;
#[cfg(feature = "benchmarks")]
#[doc(hidden)]
pub mod benchmarking;
mod engine;
mod legacy_q8;
mod limits;
mod q8_arm;
mod q8_avx2;
mod q8_engine;
mod q8_error;
mod q8_gemm;
mod tensor;

#[cfg(not(target_arch = "wasm32"))]
pub use backend::CpuModelBackendFactory;
pub use backend::{CpuBackend, CpuModelBackend, Q8CpuBackend, TextTokenizer};
pub use engine::{BatchOutput, CpuEngine};
pub use legacy_q8::{
    MarianBinaryModel, MarianTensor, MarianTensorData, MarianTensorType, Q8ValidationReport,
};
pub use marian_core::{SegmentError, TextSegment, segment_text};
pub use marian_model::{Architecture, Checksums, ModelManifest};
pub use q8_engine::{Q8CpuEngine, Q8MemoryReport};
pub use q8_error::Q8Error;
pub use q8_gemm::{
    Q8ExecutionPath, Q8Linear, Q8LinearScratch, quantize_symmetric_u8, quantize_symmetric_u8_into,
};
