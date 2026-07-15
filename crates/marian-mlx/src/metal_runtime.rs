use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::c_void,
    mem::{size_of, size_of_val},
    ptr::NonNull,
};

use half::f16;
use objc2::{AnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLanguageVersion, MTLLibrary, MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};

// objc2-metal deliberately leaves this transitive framework link to the
// application using MTLCreateSystemDefaultDevice.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

const KERNEL_SOURCE: &str = include_str!("../metal/kernels.metal");
const BUFFER_OPTIONS: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MetalStorage {
    Fp32,
    MixedF16,
}

impl MetalStorage {
    fn from_env() -> Result<Self, String> {
        match std::env::var("MARIAN_MLX_METAL_PRECISION")
            .unwrap_or_else(|_| "fp32".into())
            .as_str()
        {
            "fp32" => Ok(Self::Fp32),
            "mixed-f16" => Ok(Self::MixedF16),
            value => Err(format!(
                "unsupported MARIAN_MLX_METAL_PRECISION {value:?}; expected fp32 or mixed-f16"
            )),
        }
    }

    pub(crate) const fn code(self) -> u32 {
        match self {
            Self::Fp32 => 0,
            Self::MixedF16 => 1,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::MixedF16 => "mixed-f16",
        }
    }
}

mod private {
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
}

/// Element types that have no invalid bit patterns or padding when copied to
/// and from a Metal buffer. The private supertrait prevents new implementations
/// outside this module from weakening that guarantee.
pub(crate) trait MetalPod: private::Sealed + Copy {}

impl<T> MetalPod for T where T: private::Sealed + Copy {}

/// Inline parameter blocks copied into an MSL `constant` buffer.
///
/// # Safety
///
/// Implementors must have a stable C-compatible layout, contain only POD
/// fields, and exactly match the corresponding MSL structure.
pub(crate) unsafe trait MetalParams: Copy {}

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type RawBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type PipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type MpsMatrixKey = (usize, usize, usize);
type CachedMpsMatrix = (Retained<MPSMatrix>, RawBuffer);

#[derive(Clone)]
pub(crate) struct Buffer {
    raw: RawBuffer,
    bytes: usize,
    cacheable: bool,
}

impl Buffer {
    pub(crate) fn byte_len(&self) -> usize {
        self.bytes
    }

    pub(crate) fn read<T: MetalPod>(&self, count: usize) -> Result<Vec<T>, String> {
        let bytes = checked_bytes::<T>(count)?;
        if bytes > self.bytes {
            return Err(format!(
                "Metal buffer read requires {bytes} bytes, but allocation has {}",
                self.bytes
            ));
        }
        let pointer = self.raw.contents().cast::<T>();
        // SAFETY: StorageModeShared gives the CPU a valid mapping. The caller
        // waits for the producing command buffer before reading, and the byte
        // range was checked against the allocation above.
        Ok(unsafe { std::slice::from_raw_parts(pointer.as_ptr(), count) }.to_vec())
    }

    pub(crate) fn write<T: MetalPod>(&self, values: &[T]) -> Result<(), String> {
        let bytes = size_of_val(values);
        if bytes > self.bytes {
            return Err(format!(
                "Metal buffer write requires {bytes} bytes, but allocation has {}",
                self.bytes
            ));
        }
        // SAFETY: StorageModeShared maps this checked byte range for the CPU;
        // the scheduler serializes requests on the owning backend thread.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                self.raw.contents().as_ptr().cast::<u8>(),
                bytes,
            );
        }
        Ok(())
    }
}

pub(crate) struct Pipelines {
    pub(crate) matmul_microtile: PipelineState,
    pub(crate) matmul_bias_activation: PipelineState,
    pub(crate) embedding: PipelineState,
    pub(crate) decoder_input: PipelineState,
    pub(crate) residual_norm: PipelineState,
    pub(crate) ssru_norm: PipelineState,
    pub(crate) attention_scores: PipelineState,
    pub(crate) attention_softmax: PipelineState,
    pub(crate) attention_apply: PipelineState,
    pub(crate) attention_flash: PipelineState,
    pub(crate) output_logits: PipelineState,
    pub(crate) argmax: PipelineState,
    pub(crate) advance_decode: PipelineState,
}

pub(crate) struct MetalRuntime {
    device: Device,
    queue: Queue,
    pub(crate) pipelines: Pipelines,
    device_name: String,
    storage: MetalStorage,
    mps_descriptors: RefCell<HashMap<(usize, usize), Retained<MPSMatrixDescriptor>>>,
    mps_matmuls: RefCell<HashMap<(usize, usize, usize), Retained<MPSMatrixMultiplication>>>,
    mps_matrices: RefCell<HashMap<MpsMatrixKey, CachedMpsMatrix>>,
}

impl MetalRuntime {
    pub(crate) fn new() -> Result<Self, String> {
        let storage = MetalStorage::from_env()?;
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| "Metal has no system default device".to_string())?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| "Metal failed to create a command queue".to_string())?;

        let options = MTLCompileOptions::new();
        // This setter is deprecated only on newer SDKs; unlike MTLMathMode it
        // remains available on every macOS version supported by this project.
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        options.setLanguageVersion(MTLLanguageVersion::Version3_1);
        let source = NSString::from_str(KERNEL_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|error| format!("Metal shader compilation failed: {error:?}"))?;

        let load = |name: &'static str| -> Result<PipelineState, String> {
            let function_name = NSString::from_str(name);
            let function = library
                .newFunctionWithName(&function_name)
                .ok_or_else(|| format!("Metal shader library has no function {name}"))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| format!("Metal pipeline {name} failed: {error:?}"))
        };
        let pipelines = Pipelines {
            matmul_microtile: load("matmul_microtile_f32")?,
            matmul_bias_activation: load("matmul_bias_activation_f32")?,
            embedding: load("embedding_positions_f32")?,
            decoder_input: load("decoder_input_f32")?,
            residual_norm: load("residual_layer_norm_f32")?,
            ssru_norm: load("ssru_update_layer_norm_f32")?,
            attention_scores: load("attention_scores_f32")?,
            attention_softmax: load("attention_softmax_f32")?,
            attention_apply: load("attention_apply_f32")?,
            attention_flash: load("attention_flash_f32")?,
            output_logits: load("output_logits_f32")?,
            argmax: load("argmax_f32")?,
            advance_decode: load("advance_decode_f32")?,
        };
        let device_name = device.name().to_string();
        Ok(Self {
            device,
            queue,
            pipelines,
            device_name,
            storage,
            mps_descriptors: RefCell::new(HashMap::new()),
            mps_matmuls: RefCell::new(HashMap::new()),
            mps_matrices: RefCell::new(HashMap::new()),
        })
    }

    fn mps_descriptor(
        &self,
        rows: usize,
        columns: usize,
    ) -> Result<Retained<MPSMatrixDescriptor>, String> {
        if let Some(descriptor) = self.mps_descriptors.borrow().get(&(rows, columns)) {
            return Ok(descriptor.clone());
        }
        let row_bytes = checked_bytes::<f32>(columns)?;
        // SAFETY: The descriptor describes a packed row-major FP32 matrix.
        let descriptor = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                rows,
                columns,
                row_bytes,
                MPSDataType::Float32,
            )
        };
        self.mps_descriptors
            .borrow_mut()
            .insert((rows, columns), descriptor.clone());
        Ok(descriptor)
    }

    fn mps_matmul_kernel(
        &self,
        rows: usize,
        columns: usize,
        inner: usize,
    ) -> Retained<MPSMatrixMultiplication> {
        if let Some(kernel) = self.mps_matmuls.borrow().get(&(rows, columns, inner)) {
            return kernel.clone();
        }
        // SAFETY: The dimensions are validated against the buffers by the
        // engine before the cached shape-specific kernel is requested.
        let kernel = unsafe {
            MPSMatrixMultiplication::initWithDevice_resultRows_resultColumns_interiorColumns(
                MPSMatrixMultiplication::alloc(),
                &self.device,
                rows,
                columns,
                inner,
            )
        };
        self.mps_matmuls
            .borrow_mut()
            .insert((rows, columns, inner), kernel.clone());
        kernel
    }

    fn mps_matrix(
        &self,
        buffer: &Buffer,
        rows: usize,
        columns: usize,
    ) -> Result<Retained<MPSMatrix>, String> {
        let descriptor = self.mps_descriptor(rows, columns)?;
        if !buffer.cacheable {
            // SAFETY: The descriptor matches a packed FP32 buffer whose size
            // was checked by the engine.
            return Ok(unsafe {
                MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &buffer.raw, &descriptor)
            });
        }
        let identity = Retained::as_ptr(&buffer.raw) as *const () as usize;
        let key = (identity, rows, columns);
        if let Some((matrix, _buffer)) = self.mps_matrices.borrow().get(&key) {
            return Ok(matrix.clone());
        }
        // SAFETY: Cacheable buffers are model weights or arena allocations;
        // the retained raw buffer stored beside the view keeps them alive.
        let matrix = unsafe {
            MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &buffer.raw, &descriptor)
        };
        self.mps_matrices
            .borrow_mut()
            .insert(key, (matrix.clone(), buffer.raw.clone()));
        Ok(matrix)
    }

    pub(crate) const fn storage(&self) -> MetalStorage {
        self.storage
    }

    pub(crate) fn device_name(&self) -> &str {
        &self.device_name
    }

    pub(crate) fn empty<T: MetalPod>(&self, count: usize) -> Result<Buffer, String> {
        let bytes = checked_bytes::<T>(count)?;
        self.allocate(bytes, false)
    }

    pub(crate) fn empty_cached<T: MetalPod>(&self, count: usize) -> Result<Buffer, String> {
        let bytes = checked_bytes::<T>(count)?;
        self.allocate(bytes, true)
    }

    pub(crate) fn upload<T: MetalPod>(&self, values: &[T]) -> Result<Buffer, String> {
        if values.is_empty() {
            return Err("cannot upload an empty Metal buffer".into());
        }
        let bytes = size_of_val(values);
        let pointer = NonNull::new(values.as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| "Metal upload received a null pointer".to_string())?;
        // SAFETY: `pointer` covers `bytes` initialized bytes for the duration of
        // the call. Metal copies them into the newly allocated buffer.
        let raw = unsafe {
            self.device
                .newBufferWithBytes_length_options(pointer, bytes, BUFFER_OPTIONS)
        }
        .ok_or_else(|| format!("Metal failed to allocate and upload {bytes} bytes"))?;
        Ok(Buffer {
            raw,
            bytes,
            cacheable: false,
        })
    }

    pub(crate) fn upload_bytes(&self, values: &[u8]) -> Result<Buffer, String> {
        self.upload(values)
    }

    pub(crate) fn upload_model_f32(&self, values: &[u8]) -> Result<Buffer, String> {
        if self.storage == MetalStorage::Fp32 {
            let mut buffer = self.upload_bytes(values)?;
            buffer.cacheable = true;
            return Ok(buffer);
        }
        let chunks = values.chunks_exact(4);
        if !chunks.remainder().is_empty() {
            return Err("FP32 model tensor byte length is not divisible by four".into());
        }
        let converted = chunks
            .map(|bytes| f16::from_f32(f32::from_le_bytes(bytes.try_into().unwrap())).to_bits())
            .collect::<Vec<_>>();
        let mut buffer = self.upload(&converted)?;
        buffer.cacheable = true;
        Ok(buffer)
    }

    fn allocate(&self, bytes: usize, cacheable: bool) -> Result<Buffer, String> {
        if bytes == 0 {
            return Err("cannot allocate an empty Metal buffer".into());
        }
        let raw = self
            .device
            .newBufferWithLength_options(bytes, BUFFER_OPTIONS)
            .ok_or_else(|| format!("Metal failed to allocate {bytes} bytes"))?;
        Ok(Buffer {
            raw,
            bytes,
            cacheable,
        })
    }

    pub(crate) fn commands(&self) -> Result<Commands<'_>, String> {
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| "Metal failed to create a command buffer".to_string())?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| "Metal failed to create a compute encoder".to_string())?;
        Ok(Commands {
            command_buffer,
            encoder: RefCell::new(Some(encoder)),
            runtime: self,
            finished: false,
        })
    }
}

pub(crate) struct Commands<'a> {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: RefCell<Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>>,
    runtime: &'a MetalRuntime,
    finished: bool,
}

impl Commands<'_> {
    pub(crate) fn runtime(&self) -> &MetalRuntime {
        self.runtime
    }

    fn ensure_compute_encoder(&self) {
        if self.encoder.borrow().is_none() {
            let encoder = self
                .command_buffer
                .computeCommandEncoder()
                .expect("Metal failed to create a compute encoder");
            *self.encoder.borrow_mut() = Some(encoder);
        }
    }

    fn end_compute_encoder(&self) {
        if let Some(encoder) = self.encoder.borrow_mut().take() {
            encoder.endEncoding();
        }
    }

    pub(crate) fn dispatch<T: MetalParams>(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buffers: &[&Buffer],
        parameters: &T,
        grid: MTLSize,
        group: MTLSize,
    ) {
        self.ensure_compute_encoder();
        let encoder = self.encoder.borrow();
        let encoder = encoder
            .as_ref()
            .expect("compute encoder must be active outside MPS dispatch");
        encoder.setComputePipelineState(pipeline);
        for (index, buffer) in buffers.iter().enumerate() {
            // SAFETY: Buffers are retained by the command buffer, offsets are
            // zero, and kernel bindings are centralized in the engine.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&buffer.raw), 0, index);
            }
        }
        // SAFETY: `parameters` is a repr(C), Copy POD value in every caller and
        // remains alive until Metal has copied these inline bytes.
        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::from(parameters).cast::<c_void>(),
                size_of::<T>(),
                buffers.len(),
            );
        }
        encoder.dispatchThreads_threadsPerThreadgroup(grid, group);
    }

    pub(crate) fn dispatch_threadgroups<T: MetalParams>(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buffers: &[&Buffer],
        parameters: &T,
        groups: MTLSize,
        threads: MTLSize,
    ) {
        self.ensure_compute_encoder();
        let encoder = self.encoder.borrow();
        let encoder = encoder
            .as_ref()
            .expect("compute encoder must be active outside MPS dispatch");
        encoder.setComputePipelineState(pipeline);
        for (index, buffer) in buffers.iter().enumerate() {
            // SAFETY: See `dispatch`; this variant changes only grid semantics.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&buffer.raw), 0, index);
            }
        }
        // SAFETY: All parameter structs passed here are repr(C), Copy PODs.
        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::from(parameters).cast::<c_void>(),
                size_of::<T>(),
                buffers.len(),
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
    }

    pub(crate) fn mps_matmul(
        &self,
        lhs: &Buffer,
        rhs: &Buffer,
        output: &Buffer,
        rows: usize,
        columns: usize,
        inner: usize,
    ) -> Result<(), String> {
        self.end_compute_encoder();
        let lhs_matrix = self.runtime.mps_matrix(lhs, rows, inner)?;
        let rhs_matrix = self.runtime.mps_matrix(rhs, inner, columns)?;
        let output_matrix = self.runtime.mps_matrix(output, rows, columns)?;
        // SAFETY: Matrix descriptors exactly describe M x K, K x N, and M x N
        // packed FP32 buffers on the same Metal device.
        unsafe {
            let kernel = self.runtime.mps_matmul_kernel(rows, columns, inner);
            kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                &self.command_buffer,
                &lhs_matrix,
                &rhs_matrix,
                &output_matrix,
            );
        }
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<(), String> {
        if let Some(encoder) = self.encoder.borrow_mut().take() {
            encoder.endEncoding();
        }
        self.command_buffer.commit();
        self.command_buffer.waitUntilCompleted();
        self.finished = true;
        if let Some(error) = self.command_buffer.error() {
            return Err(format!("Metal command buffer failed: {error:?}"));
        }
        Ok(())
    }
}

impl Drop for Commands<'_> {
    fn drop(&mut self) {
        if !self.finished {
            if let Some(encoder) = self.encoder.get_mut().take() {
                encoder.endEncoding();
            }
        }
    }
}

pub(crate) const fn grid(width: usize, height: usize, depth: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

fn checked_bytes<T>(count: usize) -> Result<usize, String> {
    count
        .checked_mul(size_of::<T>())
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| "Metal buffer size is zero or overflows usize".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MatMulParams {
        rows: u32,
        cols: u32,
        inner: u32,
        has_bias: u32,
        activation: u32,
        storage: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct NormParams {
        rows: u32,
        dim: u32,
        storage: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct AttentionParams {
        batch: u32,
        query_length: u32,
        key_length: u32,
        dim: u32,
        heads: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct OutputParams {
        batch: u32,
        candidates: u32,
        dim: u32,
        storage: u32,
    }

    // SAFETY: These repr(C) u32-only layouts mirror the same structures in
    // kernels.metal; u32 has no padding or invalid bit patterns.
    unsafe impl MetalParams for MatMulParams {}
    unsafe impl MetalParams for NormParams {}
    unsafe impl MetalParams for AttentionParams {}
    unsafe impl MetalParams for OutputParams {}

    #[test]
    fn compiles_and_executes_tiled_matmul() {
        let runtime = MetalRuntime::new().unwrap();
        let lhs = runtime.upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = runtime
            .upload(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0])
            .unwrap();
        let bias = runtime.upload(&[0.5_f32, -0.5]).unwrap();
        let output = runtime.empty::<f32>(4).unwrap();
        let commands = runtime.commands().unwrap();
        commands.dispatch_threadgroups(
            &runtime.pipelines.matmul_microtile,
            &[&lhs, &rhs, &bias, &output],
            &MatMulParams {
                rows: 2,
                cols: 2,
                inner: 3,
                has_bias: 1,
                activation: 0,
                storage: 0,
            },
            grid(1, 1, 1),
            grid(32, 8, 1),
        );
        commands.finish().unwrap();
        assert_eq!(output.read::<f32>(4).unwrap(), [58.5, 63.5, 139.5, 153.5]);
    }

    #[test]
    fn executes_parallel_reductions() {
        let runtime = MetalRuntime::new().unwrap();
        let input_values = (0..384)
            .map(|index| index as f32 / 100.0 - 1.5)
            .collect::<Vec<_>>();
        let input = runtime.upload(&input_values).unwrap();
        let residual = runtime.upload(&vec![0.25_f32; 384]).unwrap();
        let scale = runtime.upload(&vec![1.0_f32; 384]).unwrap();
        let bias = runtime.upload(&vec![0.0_f32; 384]).unwrap();
        let normalized = runtime.empty::<f32>(384).unwrap();

        let scores = runtime.upload(&[1.0_f32, 2.0, 3.0]).unwrap();
        let logits = runtime
            .upload(&[4.0_f32, 9.0, 9.0, 1.0, 2.0, 7.0, 6.0, 100.0, 100.0, 100.0])
            .unwrap();
        let counts = runtime.upload(&[5_u32, 2]).unwrap();
        let selected = runtime.empty::<u32>(2).unwrap();

        let commands = runtime.commands().unwrap();
        commands.dispatch_threadgroups(
            &runtime.pipelines.residual_norm,
            &[&input, &residual, &scale, &bias, &normalized],
            &NormParams {
                rows: 1,
                dim: 384,
                storage: 0,
            },
            grid(1, 1, 1),
            grid(128, 1, 1),
        );
        commands.dispatch_threadgroups(
            &runtime.pipelines.attention_softmax,
            &[&scores],
            &AttentionParams {
                batch: 1,
                query_length: 1,
                key_length: 3,
                dim: 1,
                heads: 1,
            },
            grid(1, 1, 1),
            grid(128, 1, 1),
        );
        commands.dispatch_threadgroups(
            &runtime.pipelines.argmax,
            &[&logits, &counts, &selected],
            &OutputParams {
                batch: 2,
                candidates: 5,
                dim: 1,
                storage: 0,
            },
            grid(2, 1, 1),
            grid(128, 1, 1),
        );
        commands.finish().unwrap();

        let actual = normalized.read::<f32>(384).unwrap();
        let combined = input_values
            .iter()
            .map(|value| value + 0.25)
            .collect::<Vec<_>>();
        let mean = combined.iter().sum::<f32>() / combined.len() as f32;
        let variance = combined
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / combined.len() as f32;
        let inverse_std = (variance + 1.0e-6).sqrt().recip();
        for (index, value) in actual.iter().enumerate() {
            let expected = (combined[index] - mean) * inverse_std;
            assert!((value - expected).abs() < 2.0e-5, "index {index}");
        }

        let probabilities = scores.read::<f32>(3).unwrap();
        let denominator = 1.0 + (-1.0_f32).exp() + (-2.0_f32).exp();
        let expected = [
            (-2.0_f32).exp() / denominator,
            (-1.0_f32).exp() / denominator,
            1.0 / denominator,
        ];
        for (actual, expected) in probabilities.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        assert_eq!(selected.read::<u32>(2).unwrap(), [1, 0]);
    }

    #[test]
    fn flash_attention_matches_classic_attention_with_padding() {
        const BATCH: usize = 2;
        const QUERY: usize = 3;
        const KEY: usize = 5;
        const DIM: usize = 48;
        const HEADS: usize = 1;

        let runtime = MetalRuntime::new().unwrap();
        let query_values = (0..BATCH * QUERY * DIM)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) / 37.0)
            .collect::<Vec<_>>();
        let key_values = (0..BATCH * KEY * DIM)
            .map(|index| ((index * 29 % 113) as f32 - 56.0) / 41.0)
            .collect::<Vec<_>>();
        let value_values = (0..BATCH * KEY * DIM)
            .map(|index| ((index * 43 % 127) as f32 - 63.0) / 53.0)
            .collect::<Vec<_>>();
        let query = runtime.upload(&query_values).unwrap();
        let key = runtime.upload(&key_values).unwrap();
        let value = runtime.upload(&value_values).unwrap();
        let lengths = runtime.upload(&[5_u32, 3]).unwrap();
        let classic_scores = runtime.empty::<f32>(BATCH * HEADS * QUERY * KEY).unwrap();
        let classic_output = runtime.empty::<f32>(BATCH * QUERY * DIM).unwrap();
        let flash_output = runtime.empty::<f32>(BATCH * QUERY * DIM).unwrap();
        let params = AttentionParams {
            batch: BATCH as u32,
            query_length: QUERY as u32,
            key_length: KEY as u32,
            dim: DIM as u32,
            heads: HEADS as u32,
        };

        let commands = runtime.commands().unwrap();
        commands.dispatch(
            &runtime.pipelines.attention_scores,
            &[&query, &key, &lengths, &classic_scores],
            &params,
            grid(KEY, QUERY, BATCH * HEADS),
            grid(8, 8, 1),
        );
        commands.dispatch_threadgroups(
            &runtime.pipelines.attention_softmax,
            &[&classic_scores],
            &params,
            grid(QUERY, BATCH * HEADS, 1),
            grid(128, 1, 1),
        );
        commands.dispatch(
            &runtime.pipelines.attention_apply,
            &[&classic_scores, &value, &classic_output],
            &params,
            grid(DIM, QUERY, BATCH),
            grid(32, 4, 1),
        );
        commands.dispatch_threadgroups(
            &runtime.pipelines.attention_flash,
            &[&query, &key, &lengths, &value, &flash_output],
            &params,
            grid(QUERY.div_ceil(4), BATCH * HEADS, 1),
            grid(32, 1, 1),
        );
        commands.finish().unwrap();

        let classic = classic_output.read::<f32>(BATCH * QUERY * DIM).unwrap();
        let flash = flash_output.read::<f32>(BATCH * QUERY * DIM).unwrap();
        for (index, (classic, flash)) in classic.iter().zip(&flash).enumerate() {
            assert!(
                (classic - flash).abs() < 2.0e-5,
                "attention output {index} differs: classic={classic}, flash={flash}"
            );
        }
    }
}
