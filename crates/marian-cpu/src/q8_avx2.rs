//! Exact AVX2 kernels for Marian's asymmetric-u8 by symmetric-i8 Q8 GEMM.
//!
//! Some AVX2 int8 kernels use `vpmaddubsw`, whose intermediate result is a
//! saturating i16. Full-range Marian Q8 weights can overflow that intermediate
//! even when the final i32 dot product is representable. These kernels widen
//! both operands to i16 first and use `vpmaddwd` (`_mm256_madd_epi16`), which
//! produces non-saturating i32 pair sums.

use std::fmt;

#[cfg(target_arch = "x86_64")]
use rayon::prelude::*;

/// An error returned by an exact AVX2 Q8 operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Avx2Q8Error {
    /// The process is not running on an x86-64 CPU with AVX2 enabled.
    Unavailable,
    /// Input dimensions do not describe the supplied buffers.
    InvalidShape(String),
    /// A mathematically exact dot product does not fit in the i32 Q8 output.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    AccumulatorOverflow(i64),
}

impl fmt::Display for Avx2Q8Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => formatter.write_str("AVX2 is unavailable at runtime"),
            Self::InvalidShape(message) => formatter.write_str(message),
            Self::AccumulatorOverflow(value) => {
                write!(
                    formatter,
                    "exact Q8 accumulator {value} does not fit in i32"
                )
            }
        }
    }
}

impl std::error::Error for Avx2Q8Error {}

/// Return whether the current process may execute the exact AVX2 kernels.
///
/// This deliberately performs runtime detection. Building the crate for
/// x86-64 does not imply that the machine executing the binary supports AVX2.
#[inline]
pub(crate) fn is_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Compute an exact `(u8 - zero_point) . i8` dot product using AVX2.
#[cfg(test)]
pub(crate) fn dot_u8_i8(
    activations: &[u8],
    weights: &[i8],
    activation_zero_point: u8,
) -> Result<i32, Avx2Q8Error> {
    if activations.len() != weights.len() {
        return Err(Avx2Q8Error::InvalidShape(format!(
            "dot operands have different lengths: {} and {}",
            activations.len(),
            weights.len()
        )));
    }
    if !is_available() {
        return Err(Avx2Q8Error::Unavailable);
    }

    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `is_available` immediately above runtime-detected AVX2. The
        // slices have equal lengths, which is the only memory precondition of
        // `dot_i64_avx2`; that function bounds every vector and scalar load.
        let accumulator = unsafe { dot_i64_avx2(activations, weights, activation_zero_point) };
        i32::try_from(accumulator).map_err(|_| Avx2Q8Error::AccumulatorOverflow(accumulator))
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = activation_zero_point;
        unreachable!("non-x86 targets return before entering the AVX2 kernel")
    }
}

/// Compute row-major `[rows, inner_dim] @ [inner_dim, output_dim]` exactly.
///
/// `weights_output_input` is stored in Marian's canonical
/// `[output_dim, inner_dim]` order. The output is row-major
/// `[rows, output_dim]`. Four output channels are evaluated together so each
/// widened activation vector is reused across four independent dot products.
pub(crate) fn gemm_u8_i8(
    activations: &[u8],
    weights_output_input: &[i8],
    rows: usize,
    inner_dim: usize,
    output_dim: usize,
    activation_zero_point: u8,
) -> Result<Vec<i32>, Avx2Q8Error> {
    let expected_activations = rows.checked_mul(inner_dim).ok_or_else(|| {
        Avx2Q8Error::InvalidShape("activation dimensions overflow usize".to_owned())
    })?;
    let expected_weights = output_dim
        .checked_mul(inner_dim)
        .ok_or_else(|| Avx2Q8Error::InvalidShape("weight dimensions overflow usize".to_owned()))?;
    let output_len = rows
        .checked_mul(output_dim)
        .ok_or_else(|| Avx2Q8Error::InvalidShape("output dimensions overflow usize".to_owned()))?;
    if activations.len() != expected_activations {
        return Err(Avx2Q8Error::InvalidShape(format!(
            "activation buffer has {} values, expected {rows} x {inner_dim} = {expected_activations}",
            activations.len()
        )));
    }
    if weights_output_input.len() != expected_weights {
        return Err(Avx2Q8Error::InvalidShape(format!(
            "weight buffer has {} values, expected {output_dim} x {inner_dim} = {expected_weights}",
            weights_output_input.len()
        )));
    }
    if !is_available() {
        return Err(Avx2Q8Error::Unavailable);
    }

    #[cfg(target_arch = "x86_64")]
    {
        let mut output = vec![0_i32; output_len];
        if output_len == 0 {
            return Ok(output);
        }
        output
            .par_chunks_mut(output_dim)
            .enumerate()
            .try_for_each(|(row, output_row)| {
                let activation_start = row * inner_dim;
                let activation_row = &activations[activation_start..activation_start + inner_dim];
                let mut column = 0;

                while column + 4 <= output_dim {
                    let weight_start = column * inner_dim;
                    let weight_rows =
                        &weights_output_input[weight_start..weight_start + 4 * inner_dim];
                    // SAFETY: AVX2 was runtime-detected above. `activation_row`
                    // has `inner_dim` elements and `weight_rows` has exactly four
                    // contiguous rows of that length. The kernel bounds every
                    // vector load and handles the remaining tail scalarly.
                    let accumulators = unsafe {
                        dot4_i64_avx2(
                            activation_row,
                            weight_rows,
                            inner_dim,
                            activation_zero_point,
                        )
                    };
                    for (offset, accumulator) in accumulators.into_iter().enumerate() {
                        output_row[column + offset] = i32::try_from(accumulator)
                            .map_err(|_| Avx2Q8Error::AccumulatorOverflow(accumulator))?;
                    }
                    column += 4;
                }

                while column < output_dim {
                    let weight_start = column * inner_dim;
                    let weight_row = &weights_output_input[weight_start..weight_start + inner_dim];
                    // SAFETY: AVX2 was runtime-detected above and both slices have
                    // the same length. The kernel bounds vector and tail loads.
                    let accumulator =
                        unsafe { dot_i64_avx2(activation_row, weight_row, activation_zero_point) };
                    output_row[column] = i32::try_from(accumulator)
                        .map_err(|_| Avx2Q8Error::AccumulatorOverflow(accumulator))?;
                    column += 1;
                }
                Ok::<(), Avx2Q8Error>(())
            })?;
        Ok(output)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (output_len, activation_zero_point);
        unreachable!("non-x86 targets return before entering the AVX2 kernel")
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256i, _mm_loadu_si128, _mm256_add_epi32, _mm256_cvtepi8_epi16,
    _mm256_cvtepu8_epi16, _mm256_madd_epi16, _mm256_set1_epi16, _mm256_setzero_si256,
    _mm256_storeu_si256, _mm256_sub_epi16,
};

/// Keep each i32 vector lane below its worst-case overflow bound, then widen
/// its partial sum into an i64 scalar. One AVX2 block contributes at most two
/// products of `255 * 128` to a lane; 16,384 blocks stay below i32::MAX.
#[cfg(target_arch = "x86_64")]
const VECTOR_BLOCKS_PER_FLUSH: usize = 16_384;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i64_avx2(activations: &[u8], weights: &[i8], zero_point: u8) -> i64 {
    debug_assert_eq!(activations.len(), weights.len());
    let mut index = 0;
    let mut total = 0_i64;
    // SAFETY: the function's AVX2 target feature is guaranteed by its caller.
    let zero_points = unsafe { _mm256_set1_epi16(i16::from(zero_point)) };

    while index + 16 <= activations.len() {
        let remaining_blocks = (activations.len() - index) / 16;
        let blocks = remaining_blocks.min(VECTOR_BLOCKS_PER_FLUSH);
        let block_end = index + blocks * 16;
        // SAFETY: the function's AVX2 target feature is guaranteed by its caller.
        let mut accumulator = unsafe { _mm256_setzero_si256() };

        while index < block_end {
            // SAFETY: the loop condition and `block_end` guarantee at least 16
            // readable elements in both equal-length slices. Unaligned loads
            // are intentional and supported by the selected intrinsics.
            let activation_bytes =
                unsafe { _mm_loadu_si128(activations.as_ptr().add(index).cast::<__m128i>()) };
            // SAFETY: same 16-element bound as the activation load above.
            let weight_bytes =
                unsafe { _mm_loadu_si128(weights.as_ptr().add(index).cast::<__m128i>()) };
            // SAFETY: the function's AVX2 target feature is guaranteed by its
            // caller. Widening keeps all values within i16; madd writes i32
            // pair sums without a saturating intermediate.
            let activation_i16 =
                unsafe { _mm256_sub_epi16(_mm256_cvtepu8_epi16(activation_bytes), zero_points) };
            // SAFETY: same runtime AVX2 guarantee as above.
            let weight_i16 = unsafe { _mm256_cvtepi8_epi16(weight_bytes) };
            // SAFETY: same runtime AVX2 guarantee as above.
            let pairs = unsafe { _mm256_madd_epi16(activation_i16, weight_i16) };
            // SAFETY: same runtime AVX2 guarantee. The bounded flush interval
            // keeps each i32 lane below its worst-case overflow limit.
            accumulator = unsafe { _mm256_add_epi32(accumulator, pairs) };
            index += 16;
        }
        total += horizontal_sum_i32x8(accumulator);
    }

    for (&activation, &weight) in activations[index..].iter().zip(&weights[index..]) {
        total += (i64::from(activation) - i64::from(zero_point)) * i64::from(weight);
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot4_i64_avx2(
    activations: &[u8],
    four_weight_rows: &[i8],
    inner_dim: usize,
    zero_point: u8,
) -> [i64; 4] {
    debug_assert_eq!(activations.len(), inner_dim);
    debug_assert_eq!(four_weight_rows.len(), 4 * inner_dim);
    let mut index = 0;
    let mut totals = [0_i64; 4];
    // SAFETY: the function's AVX2 target feature is guaranteed by its caller.
    let zero_points = unsafe { _mm256_set1_epi16(i16::from(zero_point)) };

    while index + 16 <= inner_dim {
        let remaining_blocks = (inner_dim - index) / 16;
        let blocks = remaining_blocks.min(VECTOR_BLOCKS_PER_FLUSH);
        let block_end = index + blocks * 16;
        // SAFETY: the function's AVX2 target feature is guaranteed by its caller.
        let mut accumulators = [unsafe { _mm256_setzero_si256() }; 4];

        while index < block_end {
            // SAFETY: the loop bound guarantees 16 readable activation bytes.
            let activation_bytes =
                unsafe { _mm_loadu_si128(activations.as_ptr().add(index).cast::<__m128i>()) };
            // SAFETY: the function's AVX2 target feature is guaranteed by its
            // caller, and the widened subtraction remains within i16.
            let activation_i16 =
                unsafe { _mm256_sub_epi16(_mm256_cvtepu8_epi16(activation_bytes), zero_points) };
            for (weight_row, accumulator) in accumulators.iter_mut().enumerate() {
                let weight_index = weight_row * inner_dim + index;
                // SAFETY: `four_weight_rows` contains four complete
                // `inner_dim` rows and `index + 16 <= inner_dim`.
                let weight_bytes = unsafe {
                    _mm_loadu_si128(
                        four_weight_rows
                            .as_ptr()
                            .add(weight_index)
                            .cast::<__m128i>(),
                    )
                };
                // SAFETY: the function's runtime AVX2 guarantee applies to
                // these operations. Inputs are widened before the exact i32
                // pair sum, and the flush bound prevents i32 lane overflow.
                let weight_i16 = unsafe { _mm256_cvtepi8_epi16(weight_bytes) };
                // SAFETY: same runtime AVX2 guarantee as above.
                let pairs = unsafe { _mm256_madd_epi16(activation_i16, weight_i16) };
                // SAFETY: same runtime AVX2 guarantee and bounded lane sum.
                *accumulator = unsafe { _mm256_add_epi32(*accumulator, pairs) };
            }
            index += 16;
        }
        for (total, accumulator) in totals.iter_mut().zip(accumulators) {
            *total += horizontal_sum_i32x8(accumulator);
        }
    }

    for tail_index in index..inner_dim {
        let activation = i64::from(activations[tail_index]) - i64::from(zero_point);
        for weight_row in 0..4 {
            totals[weight_row] +=
                activation * i64::from(four_weight_rows[weight_row * inner_dim + tail_index]);
        }
    }
    totals
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn horizontal_sum_i32x8(vector: __m256i) -> i64 {
    let mut lanes = [0_i32; 8];
    // SAFETY: `lanes` owns exactly 32 writable bytes and the unaligned store
    // accepts any pointer alignment.
    unsafe {
        _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), vector);
    }
    lanes.into_iter().map(i64::from).sum()
}

#[cfg(test)]
mod tests {
    use super::{Avx2Q8Error, dot_u8_i8, gemm_u8_i8, is_available};

    fn value_at(index: usize) -> (u8, i8) {
        const ACTIVATIONS: [u8; 9] = [0, 1, 126, 127, 128, 253, 254, 255, 42];
        const WEIGHTS: [i8; 9] = [-128, -127, -64, -1, 0, 1, 63, 126, 127];
        (
            ACTIVATIONS[index % ACTIVATIONS.len()],
            WEIGHTS[(index * 7 + 3) % WEIGHTS.len()],
        )
    }

    fn oracle(activations: &[u8], weights: &[i8], zero_point: u8) -> i32 {
        let value = activations
            .iter()
            .zip(weights)
            .map(|(&activation, &weight)| {
                (i64::from(activation) - i64::from(zero_point)) * i64::from(weight)
            })
            .sum::<i64>();
        i32::try_from(value).unwrap()
    }

    #[test]
    fn invalid_shapes_are_rejected_before_dispatch() {
        assert_eq!(
            Avx2Q8Error::AccumulatorOverflow(i64::from(i32::MAX) + 1).to_string(),
            "exact Q8 accumulator 2147483648 does not fit in i32"
        );
        assert!(matches!(
            dot_u8_i8(&[1, 2], &[1], 127),
            Err(Avx2Q8Error::InvalidShape(_))
        ));
        assert!(matches!(
            gemm_u8_i8(&[1, 2, 3], &[1, 2], 2, 2, 1, 127),
            Err(Avx2Q8Error::InvalidShape(_))
        ));
        assert!(matches!(
            gemm_u8_i8(&[1, 2], &[1, 2, 3], 1, 2, 2, 127),
            Err(Avx2Q8Error::InvalidShape(_))
        ));
    }

    #[test]
    fn non_x86_dispatch_is_explicitly_unavailable() {
        if !is_available() {
            assert_eq!(dot_u8_i8(&[], &[], 127), Err(Avx2Q8Error::Unavailable));
            assert_eq!(
                gemm_u8_i8(&[], &[], 0, 0, 0, 127),
                Err(Avx2Q8Error::Unavailable)
            );
        }
    }

    #[test]
    fn full_range_dot_and_every_tail_match_scalar_oracle() {
        if !is_available() {
            return;
        }
        for length in [
            0, 1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 383, 384, 385, 1536, 1537,
        ] {
            let (activations, weights): (Vec<_>, Vec<_>) = (0..length).map(value_at).unzip();
            assert_eq!(
                dot_u8_i8(&activations, &weights, 127).unwrap(),
                oracle(&activations, &weights, 127),
                "length {length}"
            );
        }

        // Each adjacent pair exceeds i16::MAX before zero-point correction in
        // a `vpmaddubsw` implementation. The widened AVX2 path remains exact.
        let activations = vec![254; 31];
        let weights = vec![127; 31];
        assert_eq!(
            dot_u8_i8(&activations, &weights, 127).unwrap(),
            127 * 127 * 31
        );
    }

    #[test]
    fn tiled_gemm_matches_scalar_oracle() {
        if !is_available() {
            return;
        }
        let rows = 3;
        let inner_dim = 37;
        let output_dim = 7;
        let activations = (0..rows * inner_dim)
            .map(|index| value_at(index).0)
            .collect::<Vec<_>>();
        let weights = (0..output_dim * inner_dim)
            .map(|index| value_at(index + 11).1)
            .collect::<Vec<_>>();
        let actual = gemm_u8_i8(&activations, &weights, rows, inner_dim, output_dim, 127).unwrap();

        let mut expected = Vec::with_capacity(rows * output_dim);
        for activation_row in activations.chunks_exact(inner_dim) {
            for weight_row in weights.chunks_exact(inner_dim) {
                expected.push(oracle(activation_row, weight_row, 127));
            }
        }
        assert_eq!(actual, expected);
    }
}
