use std::{cell::RefCell, slice};

use marian_core::{TranslationBackend, TranslationInput};
use marian_cpu::{ModelManifest, Q8CpuBackend, Q8CpuEngine};
use serde::Serialize;

const MAXIMUM_WORKER_BATCH: usize = 16;

thread_local! {
    static BACKEND: RefCell<Option<Q8CpuBackend>> = const { RefCell::new(None) };
    static RESULT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[derive(Serialize)]
struct ApiOutput {
    text: String,
    input_tokens: usize,
    output_tokens: usize,
}

impl From<marian_core::TranslationOutput> for ApiOutput {
    fn from(output: marian_core::TranslationOutput) -> Self {
        Self {
            text: output.text,
            input_tokens: output.input_tokens,
            output_tokens: output.output_tokens,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc(length: usize) -> *mut u8 {
    let mut bytes = Vec::<u8>::with_capacity(length);
    let pointer = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    pointer
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc_u32(length: usize) -> *mut u32 {
    let mut words = Vec::<u32>::with_capacity(length);
    let pointer = words.as_mut_ptr();
    std::mem::forget(words);
    pointer
}

#[unsafe(no_mangle)]
/// Release an input allocation that has not been transferred to an initializer.
///
/// # Safety
///
/// `pointer` must be null or come from [`alloc`] with exactly `length` bytes of
/// capacity. The allocation must not have been released or transferred before.
pub unsafe extern "C" fn dealloc(pointer: *mut u8, length: usize) {
    if !pointer.is_null() {
        // SAFETY: The JS host only returns allocations obtained from `alloc`.
        drop(unsafe { Vec::from_raw_parts(pointer, 0, length) });
    }
}

unsafe fn take_bytes(pointer: *mut u8, length: usize) -> Vec<u8> {
    // SAFETY: Each pointer is produced by `alloc(length)`, filled exactly once,
    // and transferred to this function exactly once.
    unsafe { Vec::from_raw_parts(pointer, length, length) }
}

unsafe fn take_i8(pointer: *mut u8, length: usize) -> Vec<i8> {
    // SAFETY: u8 and i8 have identical size/alignment and the allocation came
    // from `alloc(length)` with length used as its final capacity.
    unsafe { Vec::from_raw_parts(pointer.cast::<i8>(), length, length) }
}

unsafe fn take_u32(pointer: *mut u32, length: usize) -> Vec<u32> {
    // SAFETY: The pointer came from `alloc_u32(length)` and is transferred once.
    unsafe { Vec::from_raw_parts(pointer, length, length) }
}

#[unsafe(no_mangle)]
/// Initialize the canonical Q8 backend from host-owned Wasm allocations.
///
/// # Safety
///
/// Every non-null pointer must come from [`alloc`] with the corresponding byte
/// length. Each allocation must be initialized, disjoint, and transferred
/// exactly once; this function takes ownership of all supplied allocations.
pub unsafe extern "C" fn init(
    manifest_pointer: *mut u8,
    manifest_length: usize,
    weights_pointer: *mut u8,
    weights_length: usize,
    source_pointer: *mut u8,
    source_length: usize,
    target_pointer: *mut u8,
    target_length: usize,
    shortlist_pointer: *mut u8,
    shortlist_length: usize,
) -> i32 {
    let result = Q8CpuBackend::from_preverified_owned_bytes(
        unsafe { take_bytes(manifest_pointer, manifest_length) },
        unsafe { take_bytes(weights_pointer, weights_length) },
        unsafe { take_bytes(source_pointer, source_length) },
        unsafe { take_bytes(target_pointer, target_length) },
        (shortlist_length != 0).then(|| unsafe { take_bytes(shortlist_pointer, shortlist_length) }),
    );
    match result {
        Ok(backend) => {
            BACKEND.with(|slot| *slot.borrow_mut() = Some(backend));
            set_result(b"{\"ready\":true}".to_vec());
            0
        }
        Err(error) => {
            set_error(error.to_string());
            1
        }
    }
}

/// Initialize from a Worker-specific artifact whose dense matrices were
/// already packed by the wasm32 SIMD kernel.
///
/// # Safety
///
/// Every non-null pointer must come from [`alloc`] with the corresponding byte
/// length. Each allocation must be initialized, disjoint, and transferred
/// exactly once; this function takes ownership of all supplied allocations.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_packed(
    manifest_pointer: *mut u8,
    manifest_length: usize,
    weights_pointer: *mut u8,
    weights_length: usize,
    source_pointer: *mut u8,
    source_length: usize,
    target_pointer: *mut u8,
    target_length: usize,
    shortlist_pointer: *mut u8,
    shortlist_length: usize,
) -> i32 {
    let result = Q8CpuBackend::from_preverified_worker_packed_bytes(
        unsafe { take_bytes(manifest_pointer, manifest_length) },
        unsafe { take_bytes(weights_pointer, weights_length) },
        unsafe { take_bytes(source_pointer, source_length) },
        unsafe { take_bytes(target_pointer, target_length) },
        (shortlist_length != 0).then(|| unsafe { take_bytes(shortlist_pointer, shortlist_length) }),
    );
    match result {
        Ok(backend) => {
            BACKEND.with(|slot| *slot.borrow_mut() = Some(backend));
            set_result(b"{\"ready\":true,\"packed\":true}".to_vec());
            0
        }
        Err(error) => {
            set_error(error.to_string());
            1
        }
    }
}

/// Zero-copy packed initializer. The bundle is split by the JS host and every
/// large section enters Wasm in its final owned representation.
///
/// # Safety
///
/// `dense_pointer` must come from [`alloc_u32`] with `dense_words` capacity.
/// Every other non-null pointer must come from [`alloc`] with its corresponding
/// byte length. All allocations must be initialized, disjoint, and transferred
/// exactly once; this function takes ownership of them.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn init_packed_parts(
    manifest_pointer: *mut u8,
    manifest_length: usize,
    metadata_pointer: *mut u8,
    metadata_length: usize,
    dense_pointer: *mut u32,
    dense_words: usize,
    encoder_embedding_pointer: *mut u8,
    encoder_embedding_length: usize,
    decoder_embedding_pointer: *mut u8,
    decoder_embedding_length: usize,
    source_pointer: *mut u8,
    source_length: usize,
    target_pointer: *mut u8,
    target_length: usize,
    shortlist_pointer: *mut u8,
    shortlist_length: usize,
) -> i32 {
    let result = Q8CpuBackend::from_preverified_worker_packed_parts(
        unsafe { take_bytes(manifest_pointer, manifest_length) },
        unsafe { take_bytes(metadata_pointer, metadata_length) },
        unsafe { take_u32(dense_pointer, dense_words) },
        unsafe { take_i8(encoder_embedding_pointer, encoder_embedding_length) },
        unsafe { take_i8(decoder_embedding_pointer, decoder_embedding_length) },
        unsafe { take_bytes(source_pointer, source_length) },
        unsafe { take_bytes(target_pointer, target_length) },
        (shortlist_length != 0).then(|| unsafe { take_bytes(shortlist_pointer, shortlist_length) }),
    );
    match result {
        Ok(backend) => {
            BACKEND.with(|slot| *slot.borrow_mut() = Some(backend));
            set_result(b"{\"ready\":true,\"packed\":true,\"zero_copy\":true}".to_vec());
            0
        }
        Err(error) => {
            set_error(error.to_string());
            1
        }
    }
}

/// Offline converter entry point. It must be invoked from this SIMD-enabled
/// Wasm build so the artifact is tied to the exact Worker GEMM kernel.
///
/// # Safety
///
/// Both pointers must come from [`alloc`] with the corresponding byte lengths.
/// The allocations must be initialized, disjoint, and transferred exactly
/// once; this function takes ownership of them.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pack_model(
    manifest_pointer: *mut u8,
    manifest_length: usize,
    weights_pointer: *mut u8,
    weights_length: usize,
) -> i32 {
    let manifest_bytes = unsafe { take_bytes(manifest_pointer, manifest_length) };
    let weights = unsafe { take_bytes(weights_pointer, weights_length) };
    let result = ModelManifest::from_bytes(&manifest_bytes)
        .map_err(|error| error.to_string())
        .and_then(|manifest| Q8CpuEngine::pack_worker_artifact(&weights, &manifest.architecture));
    match result {
        Ok(artifact) => {
            set_result(artifact);
            0
        }
        Err(error) => {
            set_error(error);
            1
        }
    }
}

#[unsafe(no_mangle)]
/// Translate one UTF-8 input borrowed from the host.
///
/// # Safety
///
/// `pointer` must be valid and readable for `length` bytes for the duration of
/// this call. The host retains ownership and must not mutate the allocation
/// concurrently.
pub unsafe extern "C" fn translate(
    pointer: *const u8,
    length: usize,
    max_output_tokens: usize,
) -> i32 {
    let input = unsafe { slice::from_raw_parts(pointer, length) };
    let text = match std::str::from_utf8(input) {
        Ok(text) => text,
        Err(error) => {
            set_error(format!("input is not UTF-8: {error}"));
            return 1;
        }
    };
    let result = BACKEND.with(|slot| {
        let mut slot = slot.borrow_mut();
        let backend = slot
            .as_mut()
            .ok_or_else(|| "model is not initialized".to_string())?;
        let mut input = TranslationInput::new(text, "en", "zh");
        input.max_output_tokens = max_output_tokens.clamp(1, 128);
        let output = backend
            .translate_batch(&[input])
            .map_err(|error| error.to_string())?
            .into_iter()
            .next()
            .ok_or_else(|| "backend returned no output".to_string())?;
        serde_json::to_vec(&ApiOutput::from(output)).map_err(|error| error.to_string())
    });
    match result {
        Ok(bytes) => {
            set_result(bytes);
            0
        }
        Err(error) => {
            set_error(error);
            1
        }
    }
}

/// Translate a JSON string array as one real model batch. The JS host keeps
/// the input allocation and releases it after this call returns.
///
/// # Safety
///
/// `pointer` must be valid and readable for `length` bytes for the duration of
/// this call. The host retains ownership and must not mutate the allocation
/// concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn translate_batch_json(
    pointer: *const u8,
    length: usize,
    max_output_tokens: usize,
) -> i32 {
    let bytes = unsafe { slice::from_raw_parts(pointer, length) };
    let texts = match serde_json::from_slice::<Vec<String>>(bytes) {
        Ok(texts) if !texts.is_empty() && texts.len() <= MAXIMUM_WORKER_BATCH => texts,
        Ok(texts) => {
            set_error(format!(
                "batch contains {} texts; expected 1..={MAXIMUM_WORKER_BATCH}",
                texts.len()
            ));
            return 1;
        }
        Err(error) => {
            set_error(format!("batch input is not a JSON string array: {error}"));
            return 1;
        }
    };
    if texts.iter().any(String::is_empty) {
        set_error("batch texts must be non-empty".into());
        return 1;
    }

    let result = BACKEND.with(|slot| {
        let mut slot = slot.borrow_mut();
        let backend = slot
            .as_mut()
            .ok_or_else(|| "model is not initialized".to_string())?;
        let limit = max_output_tokens.clamp(1, 128);
        let inputs = texts
            .into_iter()
            .map(|text| {
                let mut input = TranslationInput::new(text, "en", "zh");
                input.max_output_tokens = limit;
                input
            })
            .collect::<Vec<_>>();
        let outputs = backend
            .translate_batch(&inputs)
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(ApiOutput::from)
            .collect::<Vec<_>>();
        serde_json::to_vec(&outputs).map_err(|error| error.to_string())
    });
    match result {
        Ok(bytes) => {
            set_result(bytes);
            0
        }
        Err(error) => {
            set_error(error);
            1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn result_pointer() -> *const u8 {
    RESULT.with(|result| result.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn result_length() -> usize {
    RESULT.with(|result| result.borrow().len())
}

#[unsafe(no_mangle)]
#[cfg(target_arch = "wasm32")]
pub extern "C" fn memory_bytes() -> usize {
    core::arch::wasm32::memory_size(0) * 65_536
}

#[unsafe(no_mangle)]
#[cfg(not(target_arch = "wasm32"))]
pub extern "C" fn memory_bytes() -> usize {
    0
}

fn set_error(message: String) {
    let bytes = serde_json::to_vec(&serde_json::json!({ "error": message }))
        .unwrap_or_else(|_| b"{\"error\":\"serialization failed\"}".to_vec());
    set_result(bytes);
}

fn set_result(bytes: Vec<u8>) {
    RESULT.with(|result| *result.borrow_mut() = bytes);
}
