#[cfg(feature = "mlx")]
mod backend;
#[cfg(feature = "mlx")]
mod ffi;
#[cfg(feature = "mlx")]
mod manifest;

#[cfg(feature = "mlx")]
pub use backend::MlxBackend;

#[cfg(not(feature = "mlx"))]
pub struct MlxBackend;
