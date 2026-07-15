use matrixmultiply::sgemm;

const LAYER_NORM_EPSILON: f32 = 1.0e-6;

#[derive(Debug)]
pub(crate) struct Matrix {
    values: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub(crate) fn new(values: Vec<f32>, rows: usize, cols: usize) -> Result<Self, String> {
        let expected = checked_mul(rows, cols, "matrix shape")?;
        if values.len() != expected {
            return Err(format!(
                "matrix has {} elements, expected {rows} x {cols} = {expected}",
                values.len()
            ));
        }
        Ok(Self { values, rows, cols })
    }

    pub(crate) fn values(&self) -> &[f32] {
        &self.values
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }
}

pub(crate) fn matmul(
    lhs: &[f32],
    rhs: &Matrix,
    rows: usize,
    inner: usize,
    bias: Option<&Matrix>,
) -> Result<Vec<f32>, String> {
    let mut output = Vec::new();
    matmul_into(lhs, rhs, rows, inner, bias, &mut output)?;
    Ok(output)
}

pub(crate) fn matmul_into(
    lhs: &[f32],
    rhs: &Matrix,
    rows: usize,
    inner: usize,
    bias: Option<&Matrix>,
    output: &mut Vec<f32>,
) -> Result<(), String> {
    if rhs.rows != inner {
        return Err(format!(
            "matrix inner dimension mismatch: lhs has {inner}, rhs has {}",
            rhs.rows
        ));
    }
    let lhs_elements = checked_mul(rows, inner, "matrix lhs")?;
    if lhs.len() != lhs_elements {
        return Err(format!(
            "matrix lhs has {} elements, expected {lhs_elements}",
            lhs.len()
        ));
    }
    if let Some(bias) = bias {
        if bias.rows != 1 || bias.cols != rhs.cols {
            return Err(format!(
                "matrix bias has shape {} x {}, expected 1 x {}",
                bias.rows, bias.cols, rhs.cols
            ));
        }
    }

    let output_elements = checked_mul(rows, rhs.cols, "matrix output")?;
    output.clear();
    output.resize(output_elements, 0.0);
    let lhs_stride = to_isize(inner, "matrix lhs stride")?;
    let rhs_stride = to_isize(rhs.cols, "matrix rhs stride")?;
    let output_stride = to_isize(rhs.cols, "matrix output stride")?;
    // SAFETY: lhs is a contiguous rows x inner matrix, rhs is a contiguous
    // inner x cols matrix, and output is a distinct contiguous rows x cols
    // allocation. All shapes and strides were checked above.
    unsafe {
        sgemm(
            rows,
            inner,
            rhs.cols,
            1.0,
            lhs.as_ptr(),
            lhs_stride,
            1,
            rhs.values.as_ptr(),
            rhs_stride,
            1,
            0.0,
            output.as_mut_ptr(),
            output_stride,
            1,
        );
    }
    if let Some(bias) = bias {
        for row in output.chunks_exact_mut(rhs.cols) {
            for (value, &offset) in row.iter_mut().zip(&bias.values) {
                *value += offset;
            }
        }
    }
    Ok(())
}

pub(crate) fn relu_in_place(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

pub(crate) fn residual_layer_norm(
    input: &[f32],
    residual: &[f32],
    scale: &Matrix,
    bias: &Matrix,
    rows: usize,
    dim: usize,
) -> Result<Vec<f32>, String> {
    let mut output = Vec::new();
    residual_layer_norm_into(input, residual, scale, bias, rows, dim, &mut output)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_layer_norm_into(
    input: &[f32],
    residual: &[f32],
    scale: &Matrix,
    bias: &Matrix,
    rows: usize,
    dim: usize,
    output: &mut Vec<f32>,
) -> Result<(), String> {
    let elements = checked_mul(rows, dim, "layer norm shape")?;
    if input.len() != elements || residual.len() != elements {
        return Err("layer norm input shape does not match rows x dim".into());
    }
    require_vector(scale, dim, "layer norm scale")?;
    require_vector(bias, dim, "layer norm bias")?;

    output.clear();
    output.resize(elements, 0.0);
    add_slices(input, residual, output);
    for row in output.chunks_exact_mut(dim) {
        normalize_row(row, scale.values(), bias.values());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ssru_update_layer_norm(
    candidate: &[f32],
    forget_pre: &[f32],
    state: &mut [f32],
    residual: &[f32],
    scale: &Matrix,
    bias: &Matrix,
    rows: usize,
    dim: usize,
) -> Result<Vec<f32>, String> {
    let mut output = Vec::new();
    ssru_update_layer_norm_into(
        candidate,
        forget_pre,
        state,
        residual,
        scale,
        bias,
        rows,
        dim,
        &mut output,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ssru_update_layer_norm_into(
    candidate: &[f32],
    forget_pre: &[f32],
    state: &mut [f32],
    residual: &[f32],
    scale: &Matrix,
    bias: &Matrix,
    rows: usize,
    dim: usize,
    output: &mut Vec<f32>,
) -> Result<(), String> {
    let elements = checked_mul(rows, dim, "SSRU shape")?;
    if candidate.len() != elements
        || forget_pre.len() != elements
        || state.len() != elements
        || residual.len() != elements
    {
        return Err("SSRU input shape does not match rows x dim".into());
    }
    require_vector(scale, dim, "SSRU layer norm scale")?;
    require_vector(bias, dim, "SSRU layer norm bias")?;

    output.clear();
    output.resize(elements, 0.0);
    for index in 0..elements {
        let gate = 1.0 / (1.0 + (-forget_pre[index]).exp());
        let next = gate * state[index] + (1.0 - gate) * candidate[index];
        state[index] = next;
        output[index] = residual[index] + next.max(0.0);
    }
    for row in output.chunks_exact_mut(dim) {
        normalize_row(row, scale.values(), bias.values());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    lengths: &[usize],
    batch: usize,
    query_length: usize,
    key_length: usize,
    dim: usize,
    heads: usize,
) -> Result<Vec<f32>, String> {
    let mut output = Vec::new();
    let mut scores = Vec::new();
    attention_into(
        query,
        key,
        value,
        lengths,
        batch,
        query_length,
        key_length,
        dim,
        heads,
        &mut output,
        &mut scores,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_into(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    lengths: &[usize],
    batch: usize,
    query_length: usize,
    key_length: usize,
    dim: usize,
    heads: usize,
    output: &mut Vec<f32>,
    scores: &mut Vec<f32>,
) -> Result<(), String> {
    if heads == 0 || dim % heads != 0 {
        return Err(format!(
            "attention dimension {dim} is not divisible by {heads} heads"
        ));
    }
    if lengths.len() != batch
        || lengths
            .iter()
            .any(|&length| length == 0 || length > key_length)
    {
        return Err("attention lengths do not match the packed batch".into());
    }
    let query_elements = checked_product(&[batch, query_length, dim], "attention query")?;
    let key_elements = checked_product(&[batch, key_length, dim], "attention key/value")?;
    if query.len() != query_elements || key.len() != key_elements || value.len() != key_elements {
        return Err("attention tensor shape mismatch".into());
    }

    let head_dim = dim / heads;
    let attention_scale = (head_dim as f32).sqrt().recip();
    output.clear();
    output.resize(query_elements, 0.0);
    scores.clear();
    scores.resize(key_length, 0.0);
    for (batch_index, &active_keys) in lengths.iter().enumerate().take(batch) {
        for query_index in 0..query_length {
            for head in 0..heads {
                let query_base = (batch_index * query_length + query_index) * dim + head * head_dim;
                let active_scores = &mut scores[..active_keys];
                for (key_index, score) in active_scores.iter_mut().enumerate() {
                    let key_base = (batch_index * key_length + key_index) * dim + head * head_dim;
                    let mut dot = 0.0_f32;
                    for index in 0..head_dim {
                        dot += query[query_base + index] * key[key_base + index];
                    }
                    *score = dot * attention_scale;
                }
                softmax_in_place(active_scores);

                let output_base =
                    (batch_index * query_length + query_index) * dim + head * head_dim;
                for dim_index in 0..head_dim {
                    let mut attended = 0.0_f32;
                    for (key_index, &score) in active_scores.iter().enumerate() {
                        let value_index = (batch_index * key_length + key_index) * dim
                            + head * head_dim
                            + dim_index;
                        attended += score * value[value_index];
                    }
                    output[output_base + dim_index] = attended;
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn select_token(
    decoder: &[f32],
    embedding: &Matrix,
    bias: &Matrix,
    candidates: &[u32],
) -> Result<u32, String> {
    if decoder.len() != embedding.cols {
        return Err(format!(
            "decoder state has {} elements, expected {}",
            decoder.len(),
            embedding.cols
        ));
    }
    require_vector(bias, embedding.rows, "output bias")?;
    let mut best_index = 0_usize;
    let mut best_value = f32::NEG_INFINITY;
    for (candidate_index, &candidate) in candidates.iter().enumerate() {
        let token = candidate as usize;
        if token >= embedding.rows {
            return Err(format!(
                "output candidate {token} exceeds vocabulary {}",
                embedding.rows
            ));
        }
        let embedding_row = &embedding.values[token * embedding.cols..(token + 1) * embedding.cols];
        let mut logit = bias.values[token];
        for (&hidden, &weight) in decoder.iter().zip(embedding_row) {
            logit += hidden * weight;
        }
        // Strict comparison preserves the earliest (lowest candidate index)
        // on ties, matching the Metal reduction and Marian greedy decoding.
        if logit > best_value {
            best_value = logit;
            best_index = candidate_index;
        }
    }
    candidates
        .get(best_index)
        .copied()
        .ok_or_else(|| "cannot select from an empty candidate list".to_string())
}

fn require_vector(matrix: &Matrix, elements: usize, label: &str) -> Result<(), String> {
    if matrix.rows != 1 || matrix.cols != elements {
        return Err(format!(
            "{label} has shape {} x {}, expected 1 x {elements}",
            matrix.rows, matrix.cols
        ));
    }
    Ok(())
}

fn normalize_row(row: &mut [f32], scale: &[f32], bias: &[f32]) {
    let dim = row.len() as f32;
    let mean = row.iter().copied().sum::<f32>() / dim;
    let variance = row
        .iter()
        .map(|&value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f32>()
        / dim;
    let inverse_std = 1.0 / (variance + LAYER_NORM_EPSILON).sqrt();
    for ((value, &gain), &offset) in row.iter_mut().zip(scale).zip(bias) {
        *value = (*value - mean) * inverse_std * gain + offset;
    }
}

fn add_slices(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert_eq!(lhs.len(), output.len());
    let mut offset = 0;
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::{vaddq_f32, vld1q_f32, vst1q_f32};
        while offset + 4 <= lhs.len() {
            // SAFETY: The loop condition proves four readable/writable f32
            // values in each equally-sized slice. NEON is baseline AArch64.
            unsafe {
                let sum = vaddq_f32(
                    vld1q_f32(lhs.as_ptr().add(offset)),
                    vld1q_f32(rhs.as_ptr().add(offset)),
                );
                vst1q_f32(output.as_mut_ptr().add(offset), sum);
            }
            offset += 4;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was runtime-detected; the helper bounds all loads.
            offset = unsafe { add_slices_avx2(lhs, rhs, output) };
        }
    }
    for index in offset..lhs.len() {
        output[index] = lhs[index] + rhs[index];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_slices_avx2(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> usize {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};
    let mut offset = 0;
    while offset + 8 <= lhs.len() {
        // SAFETY: The loop condition proves eight readable/writable elements.
        unsafe {
            let sum = _mm256_add_ps(
                _mm256_loadu_ps(lhs.as_ptr().add(offset)),
                _mm256_loadu_ps(rhs.as_ptr().add(offset)),
            );
            _mm256_storeu_ps(output.as_mut_ptr().add(offset), sum);
        }
        offset += 8;
    }
    offset
}

fn softmax_in_place(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    let inverse_sum = 1.0 / sum;
    for value in values {
        *value *= inverse_sum;
    }
}

fn checked_product(values: &[usize], label: &str) -> Result<usize, String> {
    values.iter().try_fold(1_usize, |product, &value| {
        checked_mul(product, value, label)
    })
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize, String> {
    lhs.checked_mul(rhs)
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{label} shape {lhs} x {rhs} is zero or overflows"))
}

fn to_isize(value: usize, label: &str) -> Result<isize, String> {
    isize::try_from(value).map_err(|_| format!("{label} {value} exceeds isize"))
}

#[cfg(test)]
mod tests {
    use super::{
        Matrix, add_slices, attention, matmul, residual_layer_norm, select_token,
        ssru_update_layer_norm,
    };

    fn matrix(values: &[f32], rows: usize, cols: usize) -> Matrix {
        Matrix::new(values.to_vec(), rows, cols).unwrap()
    }

    #[test]
    fn row_major_matmul_applies_bias() {
        let lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = matrix(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let bias = matrix(&[0.5, -0.5], 1, 2);
        let actual = matmul(&lhs, &rhs, 2, 3, Some(&bias)).unwrap();
        assert_eq!(actual, vec![58.5, 63.5, 139.5, 153.5]);
    }

    #[test]
    fn simd_addition_matches_scalar_for_every_tail() {
        for length in 0..=65 {
            let lhs = (0..length)
                .map(|index| index as f32 * 0.25)
                .collect::<Vec<_>>();
            let rhs = (0..length)
                .map(|index| index as f32 * -0.5)
                .collect::<Vec<_>>();
            let mut output = vec![0.0; length];
            add_slices(&lhs, &rhs, &mut output);
            let expected = lhs.iter().zip(&rhs).map(|(a, b)| a + b).collect::<Vec<_>>();
            assert_eq!(output, expected, "length {length}");
        }
    }

    #[test]
    fn residual_norm_is_per_row() {
        let input = [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        let residual = [1.0; 8];
        let scale = matrix(&[1.0; 4], 1, 4);
        let bias = matrix(&[0.0; 4], 1, 4);
        let output = residual_layer_norm(&input, &residual, &scale, &bias, 2, 4).unwrap();
        assert!((output[0] + output[4]).abs() < 1.0e-6);
        assert!((output[3] + output[7]).abs() < 1.0e-6);
        assert!((output[..4].iter().sum::<f32>()).abs() < 1.0e-6);
        assert!((output[4..].iter().sum::<f32>()).abs() < 1.0e-6);
    }

    #[test]
    fn attention_masks_padded_keys() {
        // dim=2, one head, two keys. The second key has a much larger value,
        // but length=1 must make it completely invisible.
        let query = [1.0, 0.0];
        let key = [1.0, 0.0, 100.0, 0.0];
        let value = [3.0, 4.0, 99.0, 99.0];
        let output = attention(&query, &key, &value, &[1], 1, 1, 2, 2, 1).unwrap();
        assert_eq!(output, vec![3.0, 4.0]);
    }

    #[test]
    fn ssru_updates_persistent_state_before_normalizing() {
        let mut state = [0.0, 2.0];
        let scale = matrix(&[1.0, 1.0], 1, 2);
        let bias = matrix(&[0.0, 0.0], 1, 2);
        let output = ssru_update_layer_norm(
            &[2.0, -2.0],
            &[0.0, 0.0],
            &mut state,
            &[0.0, 0.0],
            &scale,
            &bias,
            1,
            2,
        )
        .unwrap();
        assert_eq!(state, [1.0, 0.0]);
        assert!((output[0] - 0.999_998).abs() < 1.0e-5);
        assert!((output[1] + 0.999_998).abs() < 1.0e-5);
    }

    #[test]
    fn argmax_is_stable_on_equal_logits() {
        let embedding = matrix(&[1.0, 0.0, 1.0, 0.0], 2, 2);
        let bias = matrix(&[0.0, 0.0], 1, 2);
        let selected = select_token(&[1.0, 2.0], &embedding, &bias, &[1, 0]).unwrap();
        assert_eq!(selected, 1);
    }
}
