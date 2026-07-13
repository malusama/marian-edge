use std::{
    fs,
    io::{BufReader, Read},
    path::Path,
};

use marian_core::BackendError;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Deserialize)]
pub struct ModelManifest {
    pub format: String,
    pub model_id: String,
    pub source_lang: String,
    pub target_lang: String,
    pub weights: String,
    pub source_vocab: String,
    pub target_vocab: String,
    #[serde(default)]
    pub shortlist: Option<String>,
    pub precision: String,
    pub architecture: Architecture,
    pub checksums: Checksums,
}

#[derive(Debug, Deserialize)]
pub struct Architecture {
    pub model_dim: usize,
    pub attention_heads: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub ffn_dim: usize,
    pub source_vocab_size: usize,
    pub target_vocab_size: usize,
    pub eos_id: i32,
    pub unk_id: i32,
    pub max_length_factor: usize,
}

#[derive(Debug, Deserialize)]
pub struct Checksums {
    pub weights_sha256: String,
    pub source_vocab_sha256: String,
    pub target_vocab_sha256: String,
    #[serde(default)]
    pub shortlist_sha256: Option<String>,
}

impl ModelManifest {
    pub fn load(model_dir: &Path) -> Result<Self, BackendError> {
        let path = model_dir.join("manifest.json");
        let bytes = fs::read(&path).map_err(|error| {
            BackendError::Model(format!("failed to read {}: {error}", path.display()))
        })?;
        let manifest: Self = serde_json::from_slice(&bytes)
            .map_err(|error| BackendError::Model(format!("invalid {}: {error}", path.display())))?;
        if manifest.format != "marian-mlx.transformer-ssru.v1" {
            return Err(BackendError::Model(format!(
                "unsupported model format {}",
                manifest.format
            )));
        }
        let a = &manifest.architecture;
        if (
            a.model_dim,
            a.attention_heads,
            a.encoder_layers,
            a.decoder_layers,
            a.ffn_dim,
        ) != (384, 8, 6, 4, 1536)
            || a.eos_id != 0
            || a.unk_id != 1
        {
            return Err(BackendError::Model(
                "this release supports the 384d/6e/4d SSRU graph only".into(),
            ));
        }
        Ok(manifest)
    }

    pub fn verify_runtime_files(&self, model_dir: &Path) -> Result<(), BackendError> {
        verify_sha256(
            &model_dir.join(&self.weights),
            &self.checksums.weights_sha256,
        )?;
        verify_sha256(
            &model_dir.join(&self.source_vocab),
            &self.checksums.source_vocab_sha256,
        )?;
        verify_sha256(
            &model_dir.join(&self.target_vocab),
            &self.checksums.target_vocab_sha256,
        )?;
        match (&self.shortlist, &self.checksums.shortlist_sha256) {
            (Some(path), Some(expected)) => verify_sha256(&model_dir.join(path), expected)?,
            (Some(_), None) => {
                return Err(BackendError::Model(
                    "manifest has a shortlist without shortlist_sha256".into(),
                ));
            }
            (None, Some(_)) => {
                return Err(BackendError::Model(
                    "manifest has shortlist_sha256 without a shortlist".into(),
                ));
            }
            (None, None) => {}
        }
        Ok(())
    }
}

fn verify_sha256(path: &Path, expected: &str) -> Result<(), BackendError> {
    if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(BackendError::Model(format!(
            "invalid SHA-256 in manifest for {}",
            path.display()
        )));
    }
    let file = fs::File::open(path).map_err(|error| {
        BackendError::Model(format!("failed to open {}: {error}", path.display()))
    })?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer).map_err(|error| {
            BackendError::Model(format!("failed to hash {}: {error}", path.display()))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(BackendError::Model(format!(
            "checksum mismatch for {}: expected {expected}, got {actual}",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, time::SystemTime};

    use super::*;

    #[test]
    fn runtime_file_hashes_are_enforced() {
        let unique = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "marian-mlx-manifest-{}-{unique}",
            std::process::id()
        ));
        fs::write(&path, b"abc").unwrap();

        verify_sha256(
            &path,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        .unwrap();
        assert!(verify_sha256(&path, &"0".repeat(64)).is_err());
        assert!(verify_sha256(&path, "not-a-sha256").is_err());

        fs::remove_file(path).unwrap();
    }
}
