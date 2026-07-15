//! Exact Q8 dot products optimized for Armv8.2-A dot-product hardware.
//!
//! Apple M1 and newer CPUs implement `SDOT`. The generic Q8 matrix path uses
//! rten-gemm's architecture-specific kernels, but lexical-shortlist scoring is
//! an indexed collection of independent 384-element rows and is not a regular
//! GEMM. This module keeps that sparse access pattern while vectorizing each
//! row. Integer accumulation is mathematically identical to the scalar path.

const ACTIVATION_ZERO_POINT: u8 = 127;

/// Compute `(activation_u8 - 127) . weight_i8` exactly.
///
/// Marian activation quantization produces bytes in `0..=254`, so subtracting
/// 127 in an 8-bit lane and reinterpreting it as signed yields `-127..=127`
/// without saturation or loss. On AArch64 with `dotprod`, four signed products
/// are accumulated per i32 lane by each `SDOT` instruction.
#[inline]
pub(crate) fn dot_u8_i8(activations: &[u8], weights: &[i8]) -> i32 {
    // This is a safe API around raw-pointer vector loads. Keep the length
    // check in release builds so no internal caller can make those loads
    // exceed the shorter operand.
    assert_eq!(activations.len(), weights.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: Runtime feature detection above proves SDOT support. The
            // implementation bounds every vector load to complete 16-byte
            // chunks and handles the remainder scalarly.
            return unsafe { dot_u8_i8_dotprod(activations, weights) };
        }
    }

    dot_u8_i8_scalar(activations, weights)
}

#[inline]
fn dot_u8_i8_scalar(activations: &[u8], weights: &[i8]) -> i32 {
    activations
        .iter()
        .zip(weights)
        .map(|(&activation, &weight)| {
            (i32::from(activation) - i32::from(ACTIVATION_ZERO_POINT)) * i32::from(weight)
        })
        .sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_u8_i8_dotprod(activations: &[u8], weights: &[i8]) -> i32 {
    use core::arch::aarch64::{int32x4_t, vaddq_s32, vaddvq_s32, vdupq_n_s32};

    #[target_feature(enable = "dotprod")]
    #[inline]
    unsafe fn accumulate(
        accumulator: int32x4_t,
        activation: *const u8,
        weight: *const i8,
    ) -> int32x4_t {
        use core::arch::aarch64::{vdupq_n_u8, vld1q_s8, vld1q_u8, vreinterpretq_s8_u8, vsubq_u8};
        use core::arch::asm;

        // SAFETY: The caller only passes pointers to a complete 16-byte chunk.
        let activation = unsafe { vld1q_u8(activation) };
        let centered = unsafe { vreinterpretq_s8_u8(vsubq_u8(activation, vdupq_n_u8(127))) };
        // SAFETY: The caller only passes pointers to a complete 16-byte chunk.
        let weight = unsafe { vld1q_s8(weight) };
        let mut accumulator = accumulator;
        unsafe {
            asm! {
                "sdot {acc:v}.4s, {activation:v}.16b, {weight:v}.16b",
                acc = inout(vreg) accumulator,
                activation = in(vreg) centered,
                weight = in(vreg) weight,
                options(nostack)
            }
        }
        accumulator
    }

    debug_assert_eq!(activations.len(), weights.len());
    // SAFETY: NEON is part of the baseline AArch64 ISA.
    let (mut acc0, mut acc1, mut acc2, mut acc3) = unsafe {
        (
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
        )
    };
    let mut offset = 0;

    // Four independent accumulators hide SDOT dependency latency on Apple
    // Silicon. Marian's 384-wide rows divide exactly into six iterations.
    while offset + 64 <= activations.len() {
        // SAFETY: The loop condition proves all four 16-byte chunks are in
        // bounds for both equally-sized operands.
        unsafe {
            acc0 = accumulate(
                acc0,
                activations.as_ptr().add(offset),
                weights.as_ptr().add(offset),
            );
            acc1 = accumulate(
                acc1,
                activations.as_ptr().add(offset + 16),
                weights.as_ptr().add(offset + 16),
            );
            acc2 = accumulate(
                acc2,
                activations.as_ptr().add(offset + 32),
                weights.as_ptr().add(offset + 32),
            );
            acc3 = accumulate(
                acc3,
                activations.as_ptr().add(offset + 48),
                weights.as_ptr().add(offset + 48),
            );
        }
        offset += 64;
    }
    while offset + 16 <= activations.len() {
        // SAFETY: The loop condition proves this complete vector is in bounds.
        acc0 = unsafe {
            accumulate(
                acc0,
                activations.as_ptr().add(offset),
                weights.as_ptr().add(offset),
            )
        };
        offset += 16;
    }

    // SAFETY: NEON is part of the baseline AArch64 ISA.
    let vector_sum = unsafe { vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3))) };
    vector_sum + dot_u8_i8_scalar(&activations[offset..], &weights[offset..])
}

#[cfg(test)]
mod tests {
    use super::{dot_u8_i8, dot_u8_i8_scalar};
    #[cfg(target_arch = "aarch64")]
    use crate::Q8Linear;

    #[test]
    fn every_vector_tail_matches_scalar_oracle() {
        for length in 0..=257 {
            let activations = (0..length)
                .map(|index| ((index * 73 + 19) % 255) as u8)
                .collect::<Vec<_>>();
            let weights = (0..length)
                .map(|index| ((index * 97 + 11) % 255) as i16 - 127)
                .map(|value| value as i8)
                .collect::<Vec<_>>();
            assert_eq!(
                dot_u8_i8(&activations, &weights),
                dot_u8_i8_scalar(&activations, &weights),
                "length {length}"
            );
        }
    }

    #[test]
    fn full_range_marian_row_matches_scalar_oracle() {
        let activations = (0..384)
            .map(|index| [0, 1, 126, 127, 128, 253, 254][index % 7])
            .collect::<Vec<_>>();
        let weights = (0..384)
            .map(|index| [-127, -126, -1, 0, 1, 126, 127][index % 7])
            .collect::<Vec<_>>();
        assert_eq!(
            dot_u8_i8(&activations, &weights),
            dot_u8_i8_scalar(&activations, &weights)
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn rten_selects_the_best_available_arm_kernel() {
        let linear = Q8Linear::new("probe", 64, 64, vec![1; 64 * 64], 1.0, 1.0, None)
            .expect("kernel probe should be valid");
        let expected = if std::arch::is_aarch64_feature_detected!("i8mm") {
            "i8mm"
        } else if std::arch::is_aarch64_feature_detected!("dotprod") {
            "dotprod"
        } else {
            "mlal"
        };
        eprintln!("rten Q8 kernel: {}", linear.kernel_name());
        assert!(linear.kernel_name().contains(expected));
    }
}
