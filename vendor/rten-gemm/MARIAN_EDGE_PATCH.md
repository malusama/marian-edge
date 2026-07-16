# Marian Edge patch

This directory vendors `rten-gemm 0.21.0` (MIT OR Apache-2.0) to expose a
strict packed-B reconstruction API for the Cloudflare Worker artifact format.

The patch validates the exact kernel name, matrix shape, cache-block layout,
and byte length before constructing a packed matrix. A second API permits
multiple matrices to hold checked ranges into one `Arc<Vec<u32>>`, avoiding 68
copies of the model-wide SIMD128 packed buffer. No kernel arithmetic is
changed.

Remove the `[patch.crates-io]` override when upstream provides an equivalent
safe reconstruction API.
