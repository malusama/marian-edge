use crate::MAXIMUM_POSITION;

/// Build Marian's grouped sine/cosine positional table for the supported
/// Transformer family.
pub fn sinusoidal_positions(dim: usize) -> Result<Vec<f32>, String> {
    if dim < 4 || dim % 2 != 0 {
        return Err(format!(
            "positional embedding dimension must be even and at least 4, got {dim}"
        ));
    }
    let half = dim / 2;
    let mut values = vec![0.0_f32; MAXIMUM_POSITION * dim];
    for position in 0..MAXIMUM_POSITION {
        for index in 0..half {
            let frequency = (-(index as f32) * 10_000.0_f32.ln() / (half - 1) as f32).exp();
            values[position * dim + index] = (position as f32 * frequency).sin();
            values[position * dim + half + index] = (position as f32 * frequency).cos();
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::sinusoidal_positions;

    #[test]
    fn positions_are_grouped_sin_then_cos() {
        let positions = sinusoidal_positions(384).unwrap();
        assert_eq!(positions[0], 0.0);
        assert_eq!(positions[191], 0.0);
        assert_eq!(positions[192], 1.0);
        assert_eq!(positions[383], 1.0);
        assert!((positions[384] - 1.0_f32.sin()).abs() < 1.0e-7);
        assert!((positions[384 + 192] - 1.0_f32.cos()).abs() < 1.0e-7);
        assert!(sinusoidal_positions(3).is_err());
    }
}
