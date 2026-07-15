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

use crate::MetalPrecision;

// objc2-metal deliberately leaves this transitive framework link to the
// application using MTLCreateSystemDefaultDevice.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

const KERNEL_SOURCE: &str = include_str!("../metal/kernels.metal");
const BUFFER_OPTIONS: MTLResourceOptions = MTLResourceOptions::StorageModeShared;
const MPS_SHAPE_CACHE_LIMIT: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MetalStorage {
    Fp32,
    MixedF16,
}

impl MetalStorage {
    const fn from_config(precision: MetalPrecision) -> Self {
        match precision {
            MetalPrecision::Fp32 => Self::Fp32,
            MetalPrecision::MixedF16 => Self::MixedF16,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatrixCache {
    None,
    Request,
    Permanent,
}

#[derive(Clone)]
pub(crate) struct Buffer {
    raw: RawBuffer,
    bytes: usize,
    matrix_cache: MatrixCache,
}

impl Buffer {
    pub(crate) fn byte_len(&self) -> usize {
        self.bytes
    }

    #[cfg(test)]
    pub(crate) fn identity(&self) -> usize {
        Retained::as_ptr(&self.raw) as *const () as usize
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
    pub(crate) bias_residual_norm: PipelineState,
    pub(crate) ssru_norm: PipelineState,
    pub(crate) attention_scores: PipelineState,
    pub(crate) attention_softmax: PipelineState,
    pub(crate) attention_apply: PipelineState,
    pub(crate) attention_flash: PipelineState,
    pub(crate) select_decode: PipelineState,
}

pub(crate) struct MetalRuntime {
    device: Device,
    queue: Queue,
    pub(crate) pipelines: Pipelines,
    device_name: String,
    storage: MetalStorage,
    mps_descriptors: RefCell<HashMap<(usize, usize), Retained<MPSMatrixDescriptor>>>,
    mps_matmuls: RefCell<HashMap<(usize, usize, usize), Retained<MPSMatrixMultiplication>>>,
    permanent_mps_matrices: RefCell<HashMap<MpsMatrixKey, CachedMpsMatrix>>,
    request_mps_matrices: RefCell<HashMap<MpsMatrixKey, CachedMpsMatrix>>,
}

impl MetalRuntime {
    pub(crate) fn new(precision: MetalPrecision) -> Result<Self, String> {
        let storage = MetalStorage::from_config(precision);
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
            bias_residual_norm: load("bias_residual_layer_norm_f32")?,
            ssru_norm: load("ssru_update_layer_norm_f32")?,
            attention_scores: load("attention_scores_f32")?,
            attention_softmax: load("attention_softmax_f32")?,
            attention_apply: load("attention_apply_f32")?,
            attention_flash: load("attention_flash_f32")?,
            select_decode: load("select_and_advance_decode_f32")?,
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
            permanent_mps_matrices: RefCell::new(HashMap::new()),
            request_mps_matrices: RefCell::new(HashMap::new()),
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
            .insert_bounded((rows, columns), descriptor.clone());
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
            .insert_bounded((rows, columns, inner), kernel.clone());
        kernel
    }

    fn mps_matrix(
        &self,
        buffer: &Buffer,
        rows: usize,
        columns: usize,
    ) -> Result<Retained<MPSMatrix>, String> {
        let descriptor = self.mps_descriptor(rows, columns)?;
        if buffer.matrix_cache == MatrixCache::None {
            // SAFETY: The descriptor matches a packed FP32 buffer whose size
            // was checked by the engine.
            return Ok(unsafe {
                MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &buffer.raw, &descriptor)
            });
        }
        let identity = Retained::as_ptr(&buffer.raw) as *const () as usize;
        let key = (identity, rows, columns);
        let cache = match buffer.matrix_cache {
            MatrixCache::Permanent => &self.permanent_mps_matrices,
            MatrixCache::Request => &self.request_mps_matrices,
            MatrixCache::None => unreachable!("uncached buffers return above"),
        };
        if let Some((matrix, _buffer)) = cache.borrow().get(&key) {
            return Ok(matrix.clone());
        }
        // SAFETY: Cached buffers are model weights or request-scoped arena
        // allocations; the retained raw buffer beside the view keeps them
        // alive for exactly the cache lifetime selected above.
        let matrix = unsafe {
            MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &buffer.raw, &descriptor)
        };
        cache
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

    pub(crate) fn validate_execution_plan(&self, selection_threads: usize) -> Result<(), String> {
        let device_threads = self.device.maxThreadsPerThreadgroup().width;
        let selection_limit = self.pipelines.select_decode.maxTotalThreadsPerThreadgroup();
        if selection_threads > device_threads || selection_threads > selection_limit {
            return Err(format!(
                "decode selection requests {selection_threads} threads; device/pipeline limits are {device_threads}/{selection_limit}"
            ));
        }
        let flash_width = self.pipelines.attention_flash.threadExecutionWidth();
        if flash_width != 32 {
            return Err(format!(
                "FlashAttention requires SIMD width 32, but the selected device reports {flash_width}"
            ));
        }
        let memory_limit = self.device.maxThreadgroupMemoryLength();
        for (label, pipeline) in [
            ("FlashAttention", &self.pipelines.attention_flash),
            ("decode selection", &self.pipelines.select_decode),
            ("mixed-F16 GEMM", &self.pipelines.matmul_microtile),
        ] {
            let required = pipeline.staticThreadgroupMemoryLength();
            if required > memory_limit {
                return Err(format!(
                    "{label} requires {required} bytes of threadgroup memory; device limit is {memory_limit}"
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn empty<T: MetalPod>(&self, count: usize) -> Result<Buffer, String> {
        let bytes = checked_bytes::<T>(count)?;
        self.allocate(bytes, MatrixCache::None)
    }

    pub(crate) fn empty_cached<T: MetalPod>(&self, count: usize) -> Result<Buffer, String> {
        let bytes = checked_bytes::<T>(count)?;
        self.allocate(bytes, MatrixCache::Request)
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
            matrix_cache: MatrixCache::None,
        })
    }

    pub(crate) fn upload_bytes(&self, values: &[u8]) -> Result<Buffer, String> {
        self.upload(values)
    }

    pub(crate) fn upload_model_f32(&self, values: &[u8]) -> Result<Buffer, String> {
        if self.storage == MetalStorage::Fp32 {
            let mut buffer = self.upload_bytes(values)?;
            buffer.matrix_cache = MatrixCache::Permanent;
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
        buffer.matrix_cache = MatrixCache::Permanent;
        Ok(buffer)
    }

    fn allocate(&self, bytes: usize, matrix_cache: MatrixCache) -> Result<Buffer, String> {
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
            matrix_cache,
        })
    }

    #[cfg(test)]
    pub(crate) fn commands(&self) -> Result<Commands<'_>, String> {
        self.commands_labeled("marian")
    }

    pub(crate) fn commands_labeled(&self, label: &str) -> Result<Commands<'_>, String> {
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| "Metal failed to create a command buffer".to_string())?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| "Metal failed to create a compute encoder".to_string())?;
        let label = NSString::from_str(label);
        command_buffer.setLabel(Some(&label));
        encoder.setLabel(Some(&label));
        Ok(Commands {
            command_buffer,
            encoder: RefCell::new(Some(encoder)),
            runtime: self,
            label,
            finished: false,
        })
    }

    /// Starts a new inference request and releases all MPS matrix views that
    /// retain reusable arena allocations from the previous request. Permanent
    /// model-weight views remain cached for the lifetime of the runtime.
    pub(crate) fn begin_request(&self) {
        self.request_mps_matrices.borrow_mut().clear();
    }
}

pub(crate) struct Commands<'a> {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: RefCell<Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>>,
    runtime: &'a MetalRuntime,
    label: Retained<NSString>,
    finished: bool,
}

impl Commands<'_> {
    pub(crate) fn runtime(&self) -> &MetalRuntime {
        self.runtime
    }

    fn ensure_compute_encoder(&self) -> Result<(), String> {
        if self.encoder.borrow().is_none() {
            let encoder = self
                .command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal failed to create a compute encoder".to_string())?;
            encoder.setLabel(Some(&self.label));
            *self.encoder.borrow_mut() = Some(encoder);
        }
        Ok(())
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
    ) -> Result<(), String> {
        self.ensure_compute_encoder()?;
        let encoder = self.encoder.borrow();
        let encoder = encoder
            .as_ref()
            .ok_or_else(|| "Metal compute encoder is not active".to_string())?;
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
        Ok(())
    }

    pub(crate) fn dispatch_threadgroups<T: MetalParams>(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buffers: &[&Buffer],
        parameters: &T,
        groups: MTLSize,
        threads: MTLSize,
    ) -> Result<(), String> {
        self.ensure_compute_encoder()?;
        let encoder = self.encoder.borrow();
        let encoder = encoder
            .as_ref()
            .ok_or_else(|| "Metal compute encoder is not active".to_string())?;
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
        Ok(())
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

trait BoundedShapeCache<K, V> {
    fn insert_bounded(&mut self, key: K, value: V);
}

impl<K, V> BoundedShapeCache<K, V> for HashMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    fn insert_bounded(&mut self, key: K, value: V) {
        if self.len() >= MPS_SHAPE_CACHE_LIMIT {
            self.clear();
        }
        self.insert(key, value);
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
        query_stride: u32,
        query_offset: u32,
        key_stride: u32,
        key_offset: u32,
        value_stride: u32,
        value_offset: u32,
        query_tile: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SelectDecodeParams {
        batch: u32,
        candidates: u32,
        dim: u32,
        storage: u32,
        step: u32,
        history_step: u32,
        threads: u32,
    }

    // SAFETY: These repr(C) u32-only layouts mirror the same structures in
    // kernels.metal; u32 has no padding or invalid bit patterns.
    unsafe impl MetalParams for MatMulParams {}
    unsafe impl MetalParams for NormParams {}
    unsafe impl MetalParams for AttentionParams {}
    unsafe impl MetalParams for SelectDecodeParams {}

    #[test]
    fn compiles_and_executes_tiled_matmul() {
        let runtime = MetalRuntime::new(MetalPrecision::Fp32).unwrap();
        runtime.validate_execution_plan(256).unwrap();
        let lhs = runtime.upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = runtime
            .upload(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0])
            .unwrap();
        let bias = runtime.upload(&[0.5_f32, -0.5]).unwrap();
        let output = runtime.empty::<f32>(4).unwrap();
        let commands = runtime.commands().unwrap();
        commands
            .dispatch_threadgroups(
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
            )
            .unwrap();
        commands.finish().unwrap();
        assert_eq!(output.read::<f32>(4).unwrap(), [58.5, 63.5, 139.5, 153.5]);
    }

    #[test]
    fn shape_and_request_matrix_caches_have_explicit_bounds() {
        let mut shapes = HashMap::new();
        for shape in 0..=MPS_SHAPE_CACHE_LIMIT {
            shapes.insert_bounded(shape, shape);
        }
        assert_eq!(shapes.len(), 1);

        let runtime = MetalRuntime::new(MetalPrecision::Fp32).unwrap();
        let lhs = runtime.empty_cached::<f32>(4).unwrap();
        lhs.write(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let rhs = runtime.upload(&[1.0_f32, 0.0, 0.0, 1.0]).unwrap();
        let output = runtime.empty_cached::<f32>(4).unwrap();
        let commands = runtime.commands().unwrap();
        commands.mps_matmul(&lhs, &rhs, &output, 2, 2, 2).unwrap();
        commands.finish().unwrap();
        assert_eq!(runtime.request_mps_matrices.borrow().len(), 2);
        runtime.begin_request();
        assert!(runtime.request_mps_matrices.borrow().is_empty());
    }

    #[test]
    fn executes_parallel_reductions() {
        let runtime = MetalRuntime::new(MetalPrecision::Fp32).unwrap();
        let input_values = (0..384)
            .map(|index| index as f32 / 100.0 - 1.5)
            .collect::<Vec<_>>();
        let input = runtime.upload(&input_values).unwrap();
        let linear_bias = runtime.upload(&vec![0.10_f32; 384]).unwrap();
        let residual = runtime.upload(&vec![0.15_f32; 384]).unwrap();
        let scale = runtime.upload(&vec![1.0_f32; 384]).unwrap();
        let bias = runtime.upload(&vec![0.0_f32; 384]).unwrap();
        let normalized = runtime.empty::<f32>(384).unwrap();

        let scores = runtime.upload(&[1.0_f32, 2.0, 3.0]).unwrap();
        let decoder = runtime.upload(&[1.0_f32, 1.0]).unwrap();
        let embedding = runtime
            .upload(&[4.0_f32, 9.0, 9.0, 1.0, 2.0, 7.0, 6.0, 100.0, 100.0, 100.0])
            .unwrap();
        let output_bias = runtime.upload(&[0.0_f32; 10]).unwrap();
        let candidate_ids = runtime.upload(&[0_u32, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let counts = runtime.upload(&[5_u32, 2]).unwrap();
        let previous = runtime.upload(&[-1_i32; 2]).unwrap();
        let limits = runtime.upload(&[5_u32; 2]).unwrap();
        let finished = runtime.upload(&[0_u32; 2]).unwrap();
        let history = runtime.upload(&[-1_i32; 2]).unwrap();

        let commands = runtime.commands().unwrap();
        commands
            .dispatch_threadgroups(
                &runtime.pipelines.bias_residual_norm,
                &[&input, &linear_bias, &residual, &scale, &bias, &normalized],
                &NormParams {
                    rows: 1,
                    dim: 384,
                    storage: 0,
                },
                grid(1, 1, 1),
                grid(128, 1, 1),
            )
            .unwrap();
        commands
            .dispatch_threadgroups(
                &runtime.pipelines.attention_softmax,
                &[&scores],
                &AttentionParams {
                    batch: 1,
                    query_length: 1,
                    key_length: 3,
                    dim: 1,
                    heads: 1,
                    query_stride: 1,
                    query_offset: 0,
                    key_stride: 1,
                    key_offset: 0,
                    value_stride: 1,
                    value_offset: 0,
                    query_tile: 4,
                },
                grid(1, 1, 1),
                grid(128, 1, 1),
            )
            .unwrap();
        commands
            .dispatch_threadgroups(
                &runtime.pipelines.select_decode,
                &[
                    &decoder,
                    &embedding,
                    &output_bias,
                    &candidate_ids,
                    &counts,
                    &previous,
                    &limits,
                    &finished,
                    &history,
                ],
                &SelectDecodeParams {
                    batch: 2,
                    candidates: 5,
                    dim: 1,
                    storage: 0,
                    step: 0,
                    history_step: 0,
                    threads: 256,
                },
                grid(2, 1, 1),
                grid(256, 1, 1),
            )
            .unwrap();
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
        assert_eq!(previous.read::<i32>(2).unwrap(), [1, 5]);
        assert_eq!(history.read::<i32>(2).unwrap(), [1, 5]);
        assert_eq!(finished.read::<u32>(2).unwrap(), [0, 0]);
    }

    fn assert_flash_matches_classic(
        batch: usize,
        query_length: usize,
        key_length: usize,
        dim: usize,
        heads: usize,
        query_tile: usize,
    ) {
        let runtime = MetalRuntime::new(MetalPrecision::Fp32).unwrap();
        let query_values = (0..batch * query_length * dim)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) / 37.0)
            .collect::<Vec<_>>();
        let key_values = (0..batch * key_length * dim)
            .map(|index| ((index * 29 % 113) as f32 - 56.0) / 41.0)
            .collect::<Vec<_>>();
        let value_values = (0..batch * key_length * dim)
            .map(|index| ((index * 43 % 127) as f32 - 63.0) / 53.0)
            .collect::<Vec<_>>();
        let query_stride = dim * 3;
        let query_offset = dim;
        let key_value_stride = dim * 2;
        let mut packed_query = vec![-99.0_f32; batch * query_length * query_stride];
        let mut packed_key_value = vec![-77.0_f32; batch * key_length * key_value_stride];
        for row in 0..batch * query_length {
            packed_query
                [row * query_stride + query_offset..row * query_stride + query_offset + dim]
                .copy_from_slice(&query_values[row * dim..(row + 1) * dim]);
        }
        for row in 0..batch * key_length {
            packed_key_value[row * key_value_stride..row * key_value_stride + dim]
                .copy_from_slice(&key_values[row * dim..(row + 1) * dim]);
            packed_key_value[row * key_value_stride + dim..(row + 1) * key_value_stride]
                .copy_from_slice(&value_values[row * dim..(row + 1) * dim]);
        }
        let query = runtime.upload(&packed_query).unwrap();
        let key_value = runtime.upload(&packed_key_value).unwrap();
        let length_values = (0..batch)
            .map(|row| (key_length - row.min(key_length - 1)) as u32)
            .collect::<Vec<_>>();
        let lengths = runtime.upload(&length_values).unwrap();
        let classic_scores = runtime
            .empty::<f32>(batch * heads * query_length * key_length)
            .unwrap();
        let classic_output = runtime.empty::<f32>(batch * query_length * dim).unwrap();
        let flash_output = runtime.empty::<f32>(batch * query_length * dim).unwrap();
        let params = AttentionParams {
            batch: batch as u32,
            query_length: query_length as u32,
            key_length: key_length as u32,
            dim: dim as u32,
            heads: heads as u32,
            query_stride: query_stride as u32,
            query_offset: query_offset as u32,
            key_stride: key_value_stride as u32,
            key_offset: 0,
            value_stride: key_value_stride as u32,
            value_offset: dim as u32,
            query_tile: query_tile as u32,
        };

        let commands = runtime.commands().unwrap();
        commands
            .dispatch(
                &runtime.pipelines.attention_scores,
                &[&query, &key_value, &lengths, &classic_scores],
                &params,
                grid(key_length, query_length, batch * heads),
                grid(8, 8, 1),
            )
            .unwrap();
        commands
            .dispatch_threadgroups(
                &runtime.pipelines.attention_softmax,
                &[&classic_scores],
                &params,
                grid(query_length, batch * heads, 1),
                grid(128, 1, 1),
            )
            .unwrap();
        commands
            .dispatch(
                &runtime.pipelines.attention_apply,
                &[&classic_scores, &key_value, &classic_output],
                &params,
                grid(dim, query_length, batch),
                grid(32, 4, 1),
            )
            .unwrap();
        commands
            .dispatch_threadgroups(
                &runtime.pipelines.attention_flash,
                &[&query, &key_value, &lengths, &key_value, &flash_output],
                &params,
                grid(query_length.div_ceil(query_tile), batch * heads, 1),
                grid(32, 1, 1),
            )
            .unwrap();
        commands.finish().unwrap();

        let classic = classic_output
            .read::<f32>(batch * query_length * dim)
            .unwrap();
        let flash = flash_output
            .read::<f32>(batch * query_length * dim)
            .unwrap();
        for (index, (classic, flash)) in classic.iter().zip(&flash).enumerate() {
            assert!(
                (classic - flash).abs() < 3.0e-5,
                "attention output {index} differs: classic={classic}, flash={flash}"
            );
        }
    }

    #[test]
    fn flash_attention_matches_classic_across_tiles_heads_and_key_boundaries() {
        for (batch, query, key, dim, heads, tile) in [
            (2, 1, 33, 384, 8, 1),
            (1, 4, 32, 384, 8, 2),
            (2, 5, 31, 64, 1, 4),
            (1, 5, 33, 384, 8, 4),
        ] {
            assert_flash_matches_classic(batch, query, key, dim, heads, tile);
        }
    }
}
