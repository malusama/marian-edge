use std::sync::{Arc, OnceLock};
use std::{fmt, mem::size_of};

use rten_gemm::{GemmExecutor, GemmInputA, GemmInputB, PackedBMatrix, QuantParams};
use rten_tensor::NdTensorView;

use crate::Q8Error;

const ACTIVATION_ZERO_POINT: u8 = 127;
const MIN_SYMMETRIC_Q8: f32 = -127.0;
const MAX_SYMMETRIC_Q8: f32 = 127.0;

/// The implementation selected for a [`Q8Linear`] invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q8ExecutionPath {
    /// The architecture-specific `rten-gemm` u8 x i8 -> i32 kernel.
    Rten,
    /// A non-saturating AVX2 kernel for full-range Marian Q8 weights.
    ExactAvx2,
    /// An exact Rust scalar fallback for an AVX2 saturation hazard.
    ScalarSaturationFallback,
}

/// A reusable, per-tensor symmetric Q8 linear operator.
///
/// Weights are stored in canonical `[output, input]` order. Activations are
/// quantized to signed `[-127, 127]`, represented as u8 with zero point 127,
/// and accumulated by `rten-gemm` into i32. Only the output accumulator is
/// converted back to f32; the weight matrix is never expanded to f32.
pub struct Q8Linear {
    name: String,
    input_dim: usize,
    output_dim: usize,
    weights: Vec<i8>,
    activation_quant_mult: f32,
    weight_quant_mult: f32,
    bias: Option<Vec<f32>>,
    executor: GemmExecutor<u8, i8, i32>,
    packed_weights: OnceLock<PackedBMatrix<i8>>,
    execution_path: Q8ExecutionPath,
}

/// Reusable activation and accumulator storage for [`Q8Linear::run_into`].
#[derive(Debug, Default)]
pub struct Q8LinearScratch {
    quantized: Vec<u8>,
    accumulators: Vec<i32>,
    zero_points: Vec<u8>,
}

impl fmt::Debug for Q8Linear {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Q8Linear")
            .field("name", &self.name)
            .field("input_dim", &self.input_dim)
            .field("output_dim", &self.output_dim)
            .field("activation_quant_mult", &self.activation_quant_mult)
            .field("weight_quant_mult", &self.weight_quant_mult)
            .field("kernel", &self.executor.kernel_name())
            .field("execution_path", &self.execution_path)
            .finish_non_exhaustive()
    }
}

impl Q8Linear {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        input_dim: usize,
        output_dim: usize,
        weights_output_input: Vec<i8>,
        activation_quant_mult: f32,
        weight_quant_mult: f32,
        bias: Option<Vec<f32>>,
    ) -> Result<Self, Q8Error> {
        let name = name.into();
        if input_dim == 0 || output_dim == 0 {
            return Err(Q8Error::tensor(
                &name,
                format!("linear dimensions must be positive, got {input_dim} x {output_dim}"),
            ));
        }
        let expected = input_dim.checked_mul(output_dim).ok_or_else(|| {
            Q8Error::tensor(&name, "linear dimensions overflow the address space")
        })?;
        if weights_output_input.len() != expected {
            return Err(Q8Error::tensor(
                &name,
                format!(
                    "Q8 payload has {} values, expected {output_dim} x {input_dim} = {expected}",
                    weights_output_input.len()
                ),
            ));
        }
        if weights_output_input.contains(&i8::MIN) {
            return Err(Q8Error::tensor(
                &name,
                "symmetric Marian Q8 weights must be in [-127, 127]",
            ));
        }
        validate_quant_mult(&name, "activation", activation_quant_mult)?;
        validate_quant_mult(&name, "weight", weight_quant_mult)?;
        let combined = activation_quant_mult * weight_quant_mult;
        if !combined.is_finite() || combined <= 0.0 {
            return Err(Q8Error::tensor(
                &name,
                "activation and weight quantization multipliers have an invalid product",
            ));
        }
        if let Some(bias) = &bias {
            if bias.len() != output_dim {
                return Err(Q8Error::tensor(
                    &name,
                    format!(
                        "bias has {} values, expected output dimension {output_dim}",
                        bias.len()
                    ),
                ));
            }
            if bias.iter().any(|value| !value.is_finite()) {
                return Err(Q8Error::tensor(&name, "bias contains a non-finite value"));
            }
        }

        let executor = GemmExecutor::<u8, i8, i32>::new();
        let packed_weights = OnceLock::new();
        #[cfg(not(target_arch = "aarch64"))]
        {
            let weight_view =
                NdTensorView::from_data([output_dim, input_dim], weights_output_input.as_slice());
            assert!(
                packed_weights
                    .set(executor.prepack_b(weight_view.transposed()))
                    .is_ok()
            );
        }

        // AVX2 uses vpmaddubsw internally. Restricting B to signed 7-bit makes
        // every pairwise i16 partial sum safe. Real Marian Q8 weights use the
        // full range, so preserve exactness with a scalar fallback on that one
        // kernel instead of silently returning saturated accumulators.
        let needs_saturation_fallback = executor.may_saturate()
            && weights_output_input
                .iter()
                .any(|&value| !(-64..=63).contains(&value));
        let execution_path = if needs_saturation_fallback && crate::q8_avx2::is_available() {
            Q8ExecutionPath::ExactAvx2
        } else if needs_saturation_fallback {
            Q8ExecutionPath::ScalarSaturationFallback
        } else {
            Q8ExecutionPath::Rten
        };

        #[cfg(target_arch = "wasm32")]
        let retained_weights = Vec::new();
        #[cfg(not(target_arch = "wasm32"))]
        let retained_weights = weights_output_input;

        Ok(Self {
            name,
            input_dim,
            output_dim,
            weights: retained_weights,
            activation_quant_mult,
            weight_quant_mult,
            bias,
            executor,
            packed_weights,
            execution_path,
        })
    }

    /// Rebuild a linear operator from a buffer produced by this exact
    /// `rten-gemm` kernel. This is used by the Worker artifact loader to avoid
    /// materializing canonical dense weights during cold start.
    #[allow(clippy::too_many_arguments)]
    pub fn from_packed_parts(
        name: impl Into<String>,
        input_dim: usize,
        output_dim: usize,
        activation_quant_mult: f32,
        weight_quant_mult: f32,
        bias: Option<Vec<f32>>,
        packed_kernel_name: &str,
        packed_words: Vec<u32>,
    ) -> Result<Self, Q8Error> {
        let name = name.into();
        if input_dim == 0 || output_dim == 0 {
            return Err(Q8Error::tensor(
                &name,
                format!("linear dimensions must be positive, got {input_dim} x {output_dim}"),
            ));
        }
        validate_quant_mult(&name, "activation", activation_quant_mult)?;
        validate_quant_mult(&name, "weight", weight_quant_mult)?;
        let combined = activation_quant_mult * weight_quant_mult;
        if !combined.is_finite() || combined <= 0.0 {
            return Err(Q8Error::tensor(
                &name,
                "activation and weight quantization multipliers have an invalid product",
            ));
        }
        if let Some(bias) = &bias {
            if bias.len() != output_dim {
                return Err(Q8Error::tensor(
                    &name,
                    format!(
                        "bias has {} values, expected output dimension {output_dim}",
                        bias.len()
                    ),
                ));
            }
            if bias.iter().any(|value| !value.is_finite()) {
                return Err(Q8Error::tensor(&name, "bias contains a non-finite value"));
            }
        }

        let executor = GemmExecutor::<u8, i8, i32>::new();
        let packed_weights = executor
            .restore_packed_b(packed_words, input_dim, output_dim, packed_kernel_name)
            .map_err(|error| {
                Q8Error::tensor(
                    &name,
                    format!("invalid packed weights for {packed_kernel_name}: {error}"),
                )
            })?;
        let packed_cell = OnceLock::new();
        assert!(packed_cell.set(packed_weights).is_ok());
        Ok(Self {
            name,
            input_dim,
            output_dim,
            weights: Vec::new(),
            activation_quant_mult,
            weight_quant_mult,
            bias,
            executor,
            packed_weights: packed_cell,
            execution_path: Q8ExecutionPath::Rten,
        })
    }

    /// Zero-copy variant of [`Self::from_packed_parts`] for a model-wide
    /// packed buffer shared by all dense operators.
    #[allow(clippy::too_many_arguments)]
    pub fn from_shared_packed_parts(
        name: impl Into<String>,
        input_dim: usize,
        output_dim: usize,
        activation_quant_mult: f32,
        weight_quant_mult: f32,
        bias: Option<Vec<f32>>,
        packed_kernel_name: &str,
        packed_words: Arc<Vec<u32>>,
        packed_start: usize,
        packed_len: usize,
    ) -> Result<Self, Q8Error> {
        let name = name.into();
        if input_dim == 0 || output_dim == 0 {
            return Err(Q8Error::tensor(
                &name,
                format!("linear dimensions must be positive, got {input_dim} x {output_dim}"),
            ));
        }
        validate_quant_mult(&name, "activation", activation_quant_mult)?;
        validate_quant_mult(&name, "weight", weight_quant_mult)?;
        let combined = activation_quant_mult * weight_quant_mult;
        if !combined.is_finite() || combined <= 0.0 {
            return Err(Q8Error::tensor(
                &name,
                "activation and weight quantization multipliers have an invalid product",
            ));
        }
        if let Some(bias) = &bias {
            if bias.len() != output_dim {
                return Err(Q8Error::tensor(
                    &name,
                    format!(
                        "bias has {} values, expected output dimension {output_dim}",
                        bias.len()
                    ),
                ));
            }
            if bias.iter().any(|value| !value.is_finite()) {
                return Err(Q8Error::tensor(&name, "bias contains a non-finite value"));
            }
        }
        let executor = GemmExecutor::<u8, i8, i32>::new();
        let packed_weights = executor
            .restore_shared_packed_b(
                packed_words,
                packed_start,
                packed_len,
                input_dim,
                output_dim,
                packed_kernel_name,
            )
            .map_err(|error| {
                Q8Error::tensor(
                    &name,
                    format!("invalid shared packed weights for {packed_kernel_name}: {error}"),
                )
            })?;
        let packed_cell = OnceLock::new();
        assert!(packed_cell.set(packed_weights).is_ok());
        Ok(Self {
            name,
            input_dim,
            output_dim,
            weights: Vec::new(),
            activation_quant_mult,
            weight_quant_mult,
            bias,
            executor,
            packed_weights: packed_cell,
            execution_path: Q8ExecutionPath::Rten,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn kernel_name(&self) -> &str {
        self.executor.kernel_name()
    }

    pub fn execution_path(&self) -> Q8ExecutionPath {
        self.execution_path
    }

    pub fn activation_quant_mult(&self) -> f32 {
        self.activation_quant_mult
    }

    pub fn weight_quant_mult(&self) -> f32 {
        self.weight_quant_mult
    }

    pub fn weights(&self) -> &[i8] {
        &self.weights
    }

    pub fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    /// Copies the architecture-specific packed buffer for offline artifact
    /// generation. Runtime inference never calls this method.
    pub fn packed_words(&self) -> Vec<u32> {
        self.packed_weights().clone().into_vec()
    }

    /// Return canonical and architecture-packed weight storage in bytes.
    pub fn weight_storage_bytes(&self) -> (usize, usize) {
        let canonical = self.weights.len();
        let packed = self.packed_weights.get().map_or(0, |weights| {
            weights.clone().into_vec().len() * size_of::<u32>()
        });
        (canonical, packed)
    }

    /// Compute a row-major `[rows, input_dim] @ [input_dim, output_dim]`.
    pub fn run(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, Q8Error> {
        let mut output = Vec::new();
        self.run_into(input, rows, &mut output, &mut Q8LinearScratch::default())?;
        Ok(output)
    }

    /// Compute into caller-owned storage while reusing quantization scratch.
    pub fn run_into(
        &self,
        input: &[f32],
        rows: usize,
        output: &mut Vec<f32>,
        scratch: &mut Q8LinearScratch,
    ) -> Result<(), Q8Error> {
        if rows == 0 {
            if input.is_empty() {
                output.clear();
                return Ok(());
            }
            return Err(Q8Error::tensor(
                &self.name,
                "zero rows require an empty input",
            ));
        }
        let input_len = rows.checked_mul(self.input_dim).ok_or_else(|| {
            Q8Error::tensor(&self.name, "input dimensions overflow the address space")
        })?;
        if input.len() != input_len {
            return Err(Q8Error::tensor(
                &self.name,
                format!(
                    "input has {} values, expected {rows} x {} = {input_len}",
                    input.len(),
                    self.input_dim
                ),
            ));
        }

        quantize_symmetric_u8_into(input, self.activation_quant_mult, &mut scratch.quantized)?;
        let output_len = rows.checked_mul(self.output_dim).ok_or_else(|| {
            Q8Error::tensor(&self.name, "output dimensions overflow the address space")
        })?;
        scratch.accumulators.resize(output_len, 0);
        match self.execution_path {
            Q8ExecutionPath::Rten => self.run_rten_into(
                &scratch.quantized,
                rows,
                &mut scratch.accumulators,
                &mut scratch.zero_points,
            )?,
            Q8ExecutionPath::ExactAvx2 => crate::q8_avx2::gemm_u8_i8_into(
                &scratch.quantized,
                &self.weights,
                rows,
                self.input_dim,
                self.output_dim,
                ACTIVATION_ZERO_POINT,
                &mut scratch.accumulators,
            )
            .map_err(|error| Q8Error::Gemm(format!("{}: {error}", self.name)))?,
            Q8ExecutionPath::ScalarSaturationFallback => {
                self.run_scalar_into(&scratch.quantized, rows, &mut scratch.accumulators)
            }
        }

        let inverse_scale = 1.0 / (self.activation_quant_mult * self.weight_quant_mult);
        output.clear();
        output.reserve(output_len);
        for row in scratch.accumulators.chunks_exact(self.output_dim) {
            for (column, &accumulator) in row.iter().enumerate() {
                let bias = self.bias.as_ref().map_or(0.0, |values| values[column]);
                output.push(accumulator as f32 * inverse_scale + bias);
            }
        }
        Ok(())
    }

    fn run_rten_into(
        &self,
        quantized: &[u8],
        rows: usize,
        output: &mut [i32],
        zero_points: &mut Vec<u8>,
    ) -> Result<(), Q8Error> {
        #[cfg(target_arch = "aarch64")]
        if rows <= 3 && !self.weights.is_empty() {
            output.fill(0);
            for (activations, output_row) in quantized
                .chunks_exact(self.input_dim)
                .zip(output.chunks_exact_mut(self.output_dim))
            {
                for (accumulator, weights) in output_row
                    .iter_mut()
                    .zip(self.weights.chunks_exact(self.input_dim))
                {
                    *accumulator = crate::q8_arm::dot_u8_i8(activations, weights);
                }
            }
            return Ok(());
        }

        let lhs = NdTensorView::from_data([rows, self.input_dim], quantized);
        zero_points.clear();
        zero_points.resize(rows, ACTIVATION_ZERO_POINT);
        output.fill(0);
        #[cfg(not(target_arch = "wasm32"))]
        let result = if rows == 1 && !self.weights.is_empty() {
            // rten has a specialized GEMV path for an unpacked B. Keep the
            // original Q8 bytes alongside the reusable packed matrix for it.
            let weights =
                NdTensorView::from_data([self.output_dim, self.input_dim], self.weights.as_slice());
            self.executor.gemm(
                output,
                GemmInputA::Unpacked(lhs),
                GemmInputB::Unpacked(weights.transposed()),
                1.0,
                0,
                None,
                Some(QuantParams {
                    zero_point: zero_points,
                }),
                None,
            )
        } else {
            self.executor.gemm(
                output,
                GemmInputA::Unpacked(lhs),
                GemmInputB::Packed(self.packed_weights()),
                1.0,
                0,
                None,
                Some(QuantParams {
                    zero_point: zero_points,
                }),
                None,
            )
        };
        #[cfg(target_arch = "wasm32")]
        let result = self.executor.gemm(
            output,
            GemmInputA::Unpacked(lhs),
            GemmInputB::Packed(self.packed_weights()),
            1.0,
            0,
            None,
            Some(QuantParams {
                zero_point: zero_points,
            }),
            None,
        );
        result.map_err(|error| Q8Error::Gemm(format!("{}: {error}", self.name)))?;
        Ok(())
    }

    fn packed_weights(&self) -> &PackedBMatrix<i8> {
        self.packed_weights.get_or_init(|| {
            assert!(
                !self.weights.is_empty(),
                "restored Q8 operators must include packed weights"
            );
            let weights =
                NdTensorView::from_data([self.output_dim, self.input_dim], self.weights.as_slice());
            self.executor.prepack_b(weights.transposed())
        })
    }

    fn run_scalar_into(&self, quantized: &[u8], rows: usize, output: &mut [i32]) {
        output.fill(0);
        for row in 0..rows {
            for column in 0..self.output_dim {
                let mut accumulator = 0_i32;
                let weights = &self.weights[column * self.input_dim..(column + 1) * self.input_dim];
                let activations = &quantized[row * self.input_dim..(row + 1) * self.input_dim];
                for (&activation, &weight) in activations.iter().zip(weights) {
                    accumulator += (i32::from(activation) - i32::from(ACTIVATION_ZERO_POINT))
                        * i32::from(weight);
                }
                output[row * self.output_dim + column] = accumulator;
            }
        }
    }
}

/// Quantize f32 activations using Marian's symmetric ties-to-even rule.
///
/// The returned bytes encode signed Q8 values by adding 127. This is the u8
/// representation expected by `rten-gemm` together with zero point 127.
pub fn quantize_symmetric_u8(
    input: &[f32],
    activation_quant_mult: f32,
) -> Result<Vec<u8>, Q8Error> {
    let mut output = Vec::new();
    quantize_symmetric_u8_into(input, activation_quant_mult, &mut output)?;
    Ok(output)
}

pub fn quantize_symmetric_u8_into(
    input: &[f32],
    activation_quant_mult: f32,
    output: &mut Vec<u8>,
) -> Result<(), Q8Error> {
    validate_quant_mult("activation", "activation", activation_quant_mult)?;
    output.clear();
    output.reserve(input.len());
    for (index, &value) in input.iter().enumerate() {
        if !value.is_finite() {
            return Err(Q8Error::tensor(
                "activation",
                format!("input value at index {index} is not finite"),
            ));
        }
        let signed = (value * activation_quant_mult)
            .round_ties_even()
            .clamp(MIN_SYMMETRIC_Q8, MAX_SYMMETRIC_Q8) as i16;
        output.push((signed + i16::from(ACTIVATION_ZERO_POINT)) as u8);
    }
    Ok(())
}

fn validate_quant_mult(name: &str, kind: &str, value: f32) -> Result<(), Q8Error> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Q8Error::tensor(
            name,
            format!("{kind} quantization multiplier must be finite and positive, got {value}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{Q8ExecutionPath, Q8Linear, Q8LinearScratch, quantize_symmetric_u8};

    #[test]
    fn activation_quantization_uses_ties_to_even_and_saturates() {
        let input = [
            -200.0, -127.6, -126.5, -1.5, -0.5, 0.5, 1.5, 126.5, 127.6, 200.0,
        ];
        let quantized = quantize_symmetric_u8(&input, 1.0).unwrap();
        assert_eq!(quantized, [0, 0, 1, 125, 127, 127, 129, 253, 254, 254]);
        assert!(quantize_symmetric_u8(&[f32::NAN], 1.0).is_err());
        assert!(quantize_symmetric_u8(&[0.0], 0.0).is_err());
    }

    #[test]
    fn rten_linear_matches_integer_oracle_for_gemv_and_gemm() {
        // Canonical [output, input] weights. Signed 7-bit values ensure the
        // AVX2 rten path is exact as well as the ARM dot-product paths.
        let weights = vec![2, -3, 5, 7, -11, 13, -17, 19];
        let linear = Q8Linear::new(
            "test_W",
            4,
            2,
            weights.clone(),
            4.0,
            8.0,
            Some(vec![0.25, -0.5]),
        )
        .unwrap();
        assert_eq!(linear.execution_path(), Q8ExecutionPath::Rten);

        for (input, rows) in [
            (vec![0.0, 0.25, -0.5, 1.0], 1),
            (vec![0.0, 0.25, -0.5, 1.0, 1.25, -1.0, 0.5, -0.25], 2),
            (
                vec![
                    0.0, 0.25, -0.5, 1.0, 1.25, -1.0, 0.5, -0.25, -2.0, 0.75, 1.5, 0.0,
                ],
                3,
            ),
            (
                vec![
                    0.0, 0.25, -0.5, 1.0, 1.25, -1.0, 0.5, -0.25, -2.0, 0.75, 1.5, 0.0, 0.5, 0.5,
                    -0.5, -0.5,
                ],
                4,
            ),
        ] {
            let quantized = quantize_symmetric_u8(&input, 4.0).unwrap();
            let mut expected = Vec::new();
            for row in quantized.chunks_exact(4) {
                for column in 0..2 {
                    let accumulator = row
                        .iter()
                        .zip(&weights[column * 4..(column + 1) * 4])
                        .map(|(&a, &b)| (i32::from(a) - 127) * i32::from(b))
                        .sum::<i32>();
                    expected
                        .push(accumulator as f32 / 32.0 + if column == 0 { 0.25 } else { -0.5 });
                }
            }
            assert_eq!(linear.run(&input, rows).unwrap(), expected);
        }
    }

    #[test]
    fn full_range_weights_remain_exact_on_saturating_kernels() {
        let linear = Q8Linear::new(
            "full_range_W",
            4,
            1,
            vec![127, 127, -127, -127],
            1.0,
            1.0,
            None,
        )
        .unwrap();
        let input = [127.0, 127.0, -127.0, -127.0];
        assert_eq!(linear.run(&input, 1).unwrap(), [64_516.0]);

        if linear.kernel_name().contains("avx2") {
            let expected = if crate::q8_avx2::is_available() {
                Q8ExecutionPath::ExactAvx2
            } else {
                Q8ExecutionPath::ScalarSaturationFallback
            };
            assert_eq!(linear.execution_path(), expected);
        }
    }

    #[test]
    fn constructor_and_input_shapes_are_strict() {
        assert!(Q8Linear::new("bad", 2, 2, vec![1; 3], 1.0, 1.0, None).is_err());
        assert!(Q8Linear::new("bad", 1, 1, vec![i8::MIN], 1.0, 1.0, None).is_err());
        assert!(Q8Linear::new("bad", 1, 1, vec![1], f32::NAN, 1.0, None).is_err());

        let linear = Q8Linear::new("ok", 2, 1, vec![1, 2], 1.0, 1.0, None).unwrap();
        assert!(linear.run(&[1.0], 1).is_err());
        assert!(linear.run(&[1.0], 0).is_err());
        assert!(linear.run(&[], 0).unwrap().is_empty());
    }

    #[test]
    fn shared_packed_restore_is_shape_and_kernel_checked() {
        let original = Q8Linear::new(
            "shared",
            4,
            2,
            vec![2, -3, 5, 7, -11, 13, -17, 19],
            4.0,
            8.0,
            None,
        )
        .unwrap();
        let kernel = original.kernel_name().to_owned();
        let words = Arc::new(original.packed_words());
        let restored = Q8Linear::from_shared_packed_parts(
            "shared",
            4,
            2,
            4.0,
            8.0,
            None,
            &kernel,
            Arc::clone(&words),
            0,
            words.len(),
        )
        .unwrap();
        let input = [0.0, 0.25, -0.5, 1.0];
        assert_eq!(
            restored.run(&input, 1).unwrap(),
            original.run(&input, 1).unwrap()
        );
        assert!(
            Q8Linear::from_shared_packed_parts(
                "bad-kernel",
                4,
                2,
                4.0,
                8.0,
                None,
                "not-the-selected-kernel",
                Arc::clone(&words),
                0,
                words.len(),
            )
            .is_err()
        );
        assert!(
            Q8Linear::from_shared_packed_parts(
                "bad-size",
                4,
                2,
                4.0,
                8.0,
                None,
                &kernel,
                Arc::clone(&words),
                0,
                words.len() - 1,
            )
            .is_err()
        );
    }

    #[test]
    fn run_into_reuses_all_scratch_capacity() {
        let linear = Q8Linear::new("reuse", 4, 2, vec![2; 8], 4.0, 8.0, None).unwrap();
        let mut scratch = Q8LinearScratch::default();
        let mut output = Vec::new();
        linear
            .run_into(&[0.0, 0.25, -0.5, 1.0], 1, &mut output, &mut scratch)
            .unwrap();
        let capacities = (
            scratch.quantized.capacity(),
            scratch.accumulators.capacity(),
            scratch.zero_points.capacity(),
            output.capacity(),
        );
        let expected = output.clone();
        linear
            .run_into(&[0.0, 0.25, -0.5, 1.0], 1, &mut output, &mut scratch)
            .unwrap();
        assert_eq!(output, expected);
        assert_eq!(
            capacities,
            (
                scratch.quantized.capacity(),
                scratch.accumulators.capacity(),
                scratch.zero_points.capacity(),
                output.capacity(),
            )
        );
    }
}
