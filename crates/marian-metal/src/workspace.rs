use std::{
    cell::{Cell, RefCell},
    mem::size_of,
};

use crate::metal_runtime::{Buffer, MetalPod, MetalRuntime};

/// Reusable Metal storage split by the lifetime expressed by the inference
/// graph: request data survives every decode submission, while transient data
/// may be overwritten as soon as one command buffer has completed.
#[derive(Default)]
pub(crate) struct MetalWorkspace {
    request: RefCell<BufferArena>,
    transient: RefCell<BufferArena>,
    transient_active: Cell<bool>,
}

impl MetalWorkspace {
    pub(crate) fn begin_request(&self, runtime: &MetalRuntime) -> Result<(), String> {
        if self.transient_active.get() {
            return Err("cannot begin a Metal request inside a transient frame".into());
        }
        runtime.begin_request();
        self.request.borrow_mut().rewind();
        self.transient.borrow_mut().rewind();
        Ok(())
    }

    pub(crate) fn request_upload<T: MetalPod>(
        &self,
        runtime: &MetalRuntime,
        values: &[T],
    ) -> Result<Buffer, String> {
        self.request.borrow_mut().upload(runtime, values)
    }

    pub(crate) fn request_f32(
        &self,
        runtime: &MetalRuntime,
        elements: usize,
    ) -> Result<Buffer, String> {
        self.request.borrow_mut().take::<f32>(runtime, elements)
    }

    pub(crate) fn begin_transient(&self) -> Result<TransientFrame<'_>, String> {
        if self.transient_active.replace(true) {
            return Err("nested Metal transient frames are not supported".into());
        }
        self.transient.borrow_mut().rewind();
        Ok(TransientFrame {
            active: &self.transient_active,
        })
    }

    pub(crate) fn transient<T: MetalPod>(
        &self,
        runtime: &MetalRuntime,
        elements: usize,
    ) -> Result<Buffer, String> {
        if !self.transient_active.get() {
            return Err("Metal transient allocation requires an active frame".into());
        }
        self.transient.borrow_mut().take::<T>(runtime, elements)
    }

    pub(crate) fn transient_upload<T: MetalPod>(
        &self,
        runtime: &MetalRuntime,
        values: &[T],
    ) -> Result<Buffer, String> {
        if !self.transient_active.get() {
            return Err("Metal transient upload requires an active frame".into());
        }
        self.transient.borrow_mut().upload(runtime, values)
    }

    pub(crate) fn output_f32(
        &self,
        runtime: &MetalRuntime,
        elements: usize,
    ) -> Result<Buffer, String> {
        if self.transient_active.get() {
            self.transient::<f32>(runtime, elements)
        } else {
            runtime.empty::<f32>(elements)
        }
    }
}

pub(crate) struct TransientFrame<'a> {
    active: &'a Cell<bool>,
}

impl Drop for TransientFrame<'_> {
    fn drop(&mut self) {
        self.active.set(false);
    }
}

#[derive(Default)]
struct BufferArena {
    buffers: Vec<Buffer>,
    cursor: usize,
}

impl BufferArena {
    fn rewind(&mut self) {
        self.cursor = 0;
    }

    fn take<T: MetalPod>(
        &mut self,
        runtime: &MetalRuntime,
        elements: usize,
    ) -> Result<Buffer, String> {
        let required = elements
            .checked_mul(size_of::<T>())
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| "Metal arena allocation is zero or overflows usize".to_string())?;
        let slot = self.cursor;
        self.cursor += 1;
        if slot == self.buffers.len() {
            self.buffers.push(runtime.empty_cached::<T>(elements)?);
        } else if self.buffers[slot].byte_len() < required {
            self.buffers[slot] = runtime.empty_cached::<T>(elements)?;
        }
        Ok(self.buffers[slot].clone())
    }

    fn upload<T: MetalPod>(
        &mut self,
        runtime: &MetalRuntime,
        values: &[T],
    ) -> Result<Buffer, String> {
        let buffer = self.take::<T>(runtime, values.len())?;
        buffer.write(values)?;
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalPrecision;

    #[test]
    fn transient_arena_reuses_one_history_slot_across_submissions() {
        let runtime = MetalRuntime::new(MetalPrecision::Fp32).unwrap();
        let workspace = MetalWorkspace::default();
        workspace.begin_request(&runtime).unwrap();
        let first_pointer = {
            let _frame = workspace.begin_transient().unwrap();
            let buffer = workspace.transient_upload(&runtime, &[-1_i32; 27]).unwrap();
            buffer.identity()
        };
        let second_pointer = {
            let _frame = workspace.begin_transient().unwrap();
            let buffer = workspace.transient_upload(&runtime, &[-1_i32; 18]).unwrap();
            buffer.identity()
        };
        assert_eq!(first_pointer, second_pointer);
    }
}
