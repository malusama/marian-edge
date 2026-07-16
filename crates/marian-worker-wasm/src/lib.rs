use std::{cell::RefCell, slice};

use marian_core::{TranslationBackend, TranslationInput};
use marian_cpu::Q8CpuBackend;
use serde::Serialize;

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

#[unsafe(no_mangle)]
pub extern "C" fn alloc(length: usize) -> *mut u8 {
    let mut bytes = Vec::<u8>::with_capacity(length);
    let pointer = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    pointer
}

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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
        serde_json::to_vec(&ApiOutput {
            text: output.text,
            input_tokens: output.input_tokens,
            output_tokens: output.output_tokens,
        })
        .map_err(|error| error.to_string())
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
