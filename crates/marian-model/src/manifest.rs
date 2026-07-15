use std::{
    fs,
    io::{BufReader, Read},
    path::Path,
};

use marian_core::BackendError;
use serde::Deserialize;
use sha2::{Digest, Sha256};

pub const MODEL_FORMAT_V1: &str = "marian-edge.transformer-ssru.v1";
pub const LEGACY_MODEL_FORMAT_V1: &str = "marian-mlx.transformer-ssru.v1";
pub const MAXIMUM_POSITION: usize = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformerSsruSpec {
    pub model_dim: usize,
    pub attention_heads: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub ffn_dim: usize,
    pub eos_id: i32,
    pub unk_id: i32,
    pub maximum_length_factor: usize,
}

pub const SUPPORTED_TRANSFORMER_SSRU: TransformerSsruSpec = TransformerSsruSpec {
    model_dim: 384,
    attention_heads: 8,
    encoder_layers: 6,
    decoder_layers: 4,
    ffn_dim: 1_536,
    eos_id: 0,
    unk_id: 1,
    maximum_length_factor: 8,
};

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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

impl Architecture {
    pub fn validate_supported(&self) -> Result<(), String> {
        let supported = SUPPORTED_TRANSFORMER_SSRU;
        if (
            self.model_dim,
            self.attention_heads,
            self.encoder_layers,
            self.decoder_layers,
            self.ffn_dim,
            self.eos_id,
            self.unk_id,
        ) != (
            supported.model_dim,
            supported.attention_heads,
            supported.encoder_layers,
            supported.decoder_layers,
            supported.ffn_dim,
            supported.eos_id,
            supported.unk_id,
        ) {
            return Err(format!(
                "supported Transformer-SSRU graph is {}d/{}h/{}e/{}d/{}ffn with EOS={} and UNK={}",
                supported.model_dim,
                supported.attention_heads,
                supported.encoder_layers,
                supported.decoder_layers,
                supported.ffn_dim,
                supported.eos_id,
                supported.unk_id,
            ));
        }
        if self.source_vocab_size <= 2 || self.target_vocab_size <= 2 {
            return Err("model vocabulary sizes must contain EOS, UNK, and warmup tokens".into());
        }
        if !(1..=supported.maximum_length_factor).contains(&self.max_length_factor) {
            return Err(format!(
                "model max_length_factor must be between 1 and {}",
                supported.maximum_length_factor
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
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
        manifest.validate()?;
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

    fn validate(&self) -> Result<(), BackendError> {
        if !matches!(
            self.format.as_str(),
            MODEL_FORMAT_V1 | LEGACY_MODEL_FORMAT_V1
        ) {
            return Err(BackendError::Model(format!(
                "unsupported model format {}",
                self.format
            )));
        }
        if !matches!(self.precision.as_str(), "fp32" | "q8") {
            return Err(BackendError::Model(format!(
                "unsupported model precision {}; expected fp32 or q8",
                self.precision
            )));
        }
        self.architecture
            .validate_supported()
            .map_err(BackendError::Model)?;
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

    fn manifest(precision: &str, max_length_factor: usize) -> ModelManifest {
        ModelManifest {
            format: MODEL_FORMAT_V1.into(),
            model_id: "test".into(),
            source_lang: "en".into(),
            target_lang: "zh".into(),
            weights: "model.safetensors".into(),
            source_vocab: "source.spm".into(),
            target_vocab: "target.spm".into(),
            shortlist: None,
            precision: precision.into(),
            architecture: Architecture {
                model_dim: 384,
                attention_heads: 8,
                encoder_layers: 6,
                decoder_layers: 4,
                ffn_dim: 1536,
                source_vocab_size: 32_000,
                target_vocab_size: 32_000,
                eos_id: 0,
                unk_id: 1,
                max_length_factor,
            },
            checksums: Checksums {
                weights_sha256: String::new(),
                source_vocab_sha256: String::new(),
                target_vocab_sha256: String::new(),
                shortlist_sha256: None,
            },
        }
    }

    #[test]
    fn supported_precisions_and_bounded_length_factor_are_enforced() {
        assert!(manifest("fp32", 1).validate().is_ok());
        assert!(manifest("fp32", 8).validate().is_ok());
        assert!(manifest("q8", 3).validate().is_ok());
        assert!(manifest("fp16", 3).validate().is_err());
        assert!(manifest("fp32", 0).validate().is_err());
        assert!(manifest("fp32", 9).validate().is_err());
    }

    #[test]
    fn legacy_model_format_remains_loadable_during_rename() {
        let mut value = manifest("fp32", 3);
        value.format = LEGACY_MODEL_FORMAT_V1.into();
        assert!(value.validate().is_ok());
    }

    #[test]
    fn vocabularies_must_contain_required_special_tokens() {
        let mut value = manifest("fp32", 3);
        value.architecture.source_vocab_size = 2;
        assert!(value.validate().is_err());

        let mut value = manifest("fp32", 3);
        value.architecture.target_vocab_size = 2;
        assert!(value.validate().is_err());
    }

    #[test]
    fn runtime_file_hashes_are_enforced() {
        let unique = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "marian-model-manifest-{}-{unique}",
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
