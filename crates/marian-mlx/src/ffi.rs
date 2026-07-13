#[cxx::bridge(namespace = "marian_mlx")]
pub(crate) mod bridge {
    struct BatchOutput {
        tokens: Vec<i32>,
        offsets: Vec<u32>,
        scores: Vec<f32>,
    }

    unsafe extern "C++" {
        include!("marian-mlx/native/include/engine.hpp");

        type Engine;

        #[allow(dead_code)]
        fn validate_shortlist(shortlist_path: &str) -> Result<()>;

        fn new_engine(
            weights_path: &str,
            shortlist_path: &str,
            metallib_path: &str,
            max_length_factor: usize,
        ) -> Result<UniquePtr<Engine>>;

        fn translate(
            self: Pin<&mut Engine>,
            tokens: &[i32],
            offsets: &[u32],
            max_output_tokens: usize,
        ) -> Result<BatchOutput>;

        fn warmup(self: Pin<&mut Engine>) -> Result<()>;
        fn device_name(self: &Engine) -> String;
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, time::SystemTime};

    use super::bridge;

    const SHORTLIST_MAGIC: u64 = 17_373_278_592_220_534_773;
    const OFFSET_COUNT: usize = 32_001;

    fn write_shortlist(offsets: &[u64]) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "marian-mlx-shortlist-{}-{unique}.bin",
            std::process::id()
        ));
        let mut bytes = Vec::with_capacity(48 + offsets.len() * 8 + 8);
        for value in [SHORTLIST_MAGIC, 0, 0, 0, offsets.len() as u64, 2] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for offset in offsets {
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn shortlist_rejects_non_monotonic_and_out_of_range_offsets() {
        let mut offsets = vec![2; OFFSET_COUNT];
        offsets[0] = 0;
        offsets[2] = 1;
        let path = write_shortlist(&offsets);
        let error = bridge::validate_shortlist(path.to_str().unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("not monotonic"), "{error}");
        fs::remove_file(path).unwrap();

        offsets[2] = 2;
        offsets[1] = 3;
        let path = write_shortlist(&offsets);
        let error = bridge::validate_shortlist(path.to_str().unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside its target body"), "{error}");
        fs::remove_file(path).unwrap();
    }
}
