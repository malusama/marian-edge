use std::{collections::HashSet, env, fs, path::Path};

use marian_cpu::{MarianBinaryModel, MarianTensorData, MarianTensorType, Q8ValidationReport};
use safetensors::{Dtype, SafeTensors};

const EXPECTED_Q8_TENSORS: usize = 253;
const EXPECTED_FP32_TENSORS: usize = 183;
const EXPECTED_Q8_WEIGHTS: usize = 70;
const MAXIMUM_FP32_FILE_BYTES: u64 = 256 * 1024 * 1024;

fn required_path(variable: &str) -> String {
    env::var(variable).unwrap_or_else(|_| {
        panic!(
            "{variable} must name the matching real Marian artifact; this ignored test never downloads models"
        )
    })
}

fn read_bounded(path: &Path) -> Vec<u8> {
    let metadata = fs::metadata(path)
        .unwrap_or_else(|error| panic!("failed to inspect {}: {error}", path.display()));
    assert!(
        metadata.len() <= MAXIMUM_FP32_FILE_BYTES,
        "FP32 artifact {} is {} bytes; test maximum is {MAXIMUM_FP32_FILE_BYTES}",
        path.display(),
        metadata.len()
    );
    fs::read(path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn decode_f32(bytes: &[u8], name: &str) -> Vec<f32> {
    assert_eq!(
        bytes.len() % size_of::<f32>(),
        0,
        "FP32 tensor {name} has a truncated payload"
    );
    bytes
        .chunks_exact(size_of::<f32>())
        .enumerate()
        .map(|(index, bytes)| {
            let value = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
            assert!(
                value.is_finite(),
                "FP32 tensor {name} contains non-finite value at element {index}"
            );
            value
        })
        .collect()
}

#[test]
#[ignore = "set MARIAN_Q8_MODEL and MARIAN_FP32_MODEL to matching real artifacts"]
fn q8_artifact_is_the_exact_quantization_of_fp32_graph() {
    let q8_path = required_path("MARIAN_Q8_MODEL");
    let fp32_path = required_path("MARIAN_FP32_MODEL");
    let q8 = MarianBinaryModel::open(&q8_path)
        .unwrap_or_else(|error| panic!("failed to load Q8 artifact {q8_path}: {error}"));
    assert_eq!(q8.len(), EXPECTED_Q8_TENSORS);
    assert_eq!(
        q8.validate_q8().expect("Q8 cross-tensor validation"),
        Q8ValidationReport {
            dense_linears: 68,
            embeddings: 2,
            activation_scales: 69,
        }
    );

    let fp32_bytes = read_bounded(Path::new(&fp32_path));
    let fp32 = SafeTensors::deserialize(&fp32_bytes)
        .unwrap_or_else(|error| panic!("failed to load FP32 artifact {fp32_path}: {error}"));
    let fp32_tensors = fp32.tensors();
    assert_eq!(fp32_tensors.len(), EXPECTED_FP32_TENSORS);

    let fp32_names = fp32_tensors
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<HashSet<_>>();
    assert_eq!(fp32_names.len(), EXPECTED_FP32_TENSORS);

    let mut q8_weight_count = 0_usize;
    let mut maximum_reconstruction_error = 0.0_f32;
    let mut maximum_error_name = String::new();

    for (name, reference) in fp32_tensors {
        assert_eq!(
            reference.dtype(),
            Dtype::F32,
            "reference tensor {name} is not FP32"
        );
        let tensor = q8.tensor(&name).unwrap_or_else(|error| {
            panic!("Q8 artifact is missing reference tensor {name}: {error}")
        });
        assert_eq!(
            tensor.shape(),
            reference.shape(),
            "shape differs for tensor {name}"
        );
        let reference = decode_f32(reference.data(), &name);

        match tensor.data() {
            MarianTensorData::Float32(actual) => {
                assert_eq!(actual.len(), reference.len(), "length differs for {name}");
                for (index, (&actual, &reference)) in actual.iter().zip(&reference).enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        reference.to_bits(),
                        "FP32 constant {name} differs at element {index}"
                    );
                }
            }
            MarianTensorData::Intgemm8 { values, quant_mult } => {
                q8_weight_count += 1;
                assert_eq!(tensor.shape().len(), 2, "Q8 weight {name} is not rank 2");
                let rows = tensor.shape()[0];
                let columns = tensor.shape()[1];
                assert_eq!(values.len(), rows * columns, "Q8 payload length for {name}");
                let embedding_layout = name.ends_with("_Wemb");

                for row in 0..rows {
                    for column in 0..columns {
                        let reference_index = row * columns + column;
                        // Dense records keep the logical [input, output] header,
                        // but their architecture-independent bytes are already
                        // transposed to [output, input]. Wemb remains row-major.
                        let quantized_index = if embedding_layout {
                            reference_index
                        } else {
                            column * rows + row
                        };
                        let expected = (reference[reference_index] * quant_mult)
                            .round_ties_even()
                            .clamp(-127.0, 127.0) as i8;
                        assert_eq!(
                            values[quantized_index], expected,
                            "Q8 value differs for {name} at logical [{row}, {column}]"
                        );

                        let reconstructed = f32::from(values[quantized_index]) / quant_mult;
                        let error = (reference[reference_index] - reconstructed).abs();
                        let rounding_bound = 0.500_001_f32 / quant_mult;
                        assert!(
                            error <= rounding_bound,
                            "Q8 reconstruction error {error} exceeds {rounding_bound} for {name} at logical [{row}, {column}]"
                        );
                        if error > maximum_reconstruction_error {
                            maximum_reconstruction_error = error;
                            maximum_error_name.clone_from(&name);
                        }
                    }
                }
            }
            MarianTensorData::Int8(_) => {
                panic!("reference tensor {name} unexpectedly became raw int8")
            }
        }
    }
    assert_eq!(q8_weight_count, EXPECTED_Q8_WEIGHTS);

    let extras = q8
        .tensors()
        .iter()
        .filter(|tensor| !fp32_names.contains(tensor.name()))
        .collect::<Vec<_>>();
    assert_eq!(extras.len(), 70);
    assert_eq!(
        extras
            .iter()
            .filter(|tensor| tensor.name().ends_with("_QuantMultA"))
            .count(),
        69
    );
    let config = extras
        .iter()
        .find(|tensor| tensor.name() == "special:model.yml")
        .expect("Q8 artifact has no embedded Marian graph config");
    assert_eq!(config.tensor_type(), MarianTensorType::Int8);

    eprintln!(
        "validated {EXPECTED_FP32_TENSORS} tensors and {EXPECTED_Q8_WEIGHTS} exact quantizations; maximum reconstruction error {maximum_reconstruction_error:.9} in {maximum_error_name}"
    );
}
