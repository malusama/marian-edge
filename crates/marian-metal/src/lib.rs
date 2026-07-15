#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod backend;
mod config;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod engine;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod metal_runtime;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod tuning;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod workspace;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub use backend::{MetalBackend, MlxBackend};

#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
pub struct MetalBackend;
#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
pub struct MlxBackend;
pub use config::{MetalAttention, MetalConfig, MetalPrecision, MetalProfile};
