use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::SystemTime,
};

use marian_core::{TranslationBackend, TranslationInput};
use marian_cpu::{Q8CpuBackend, Q8CpuEngine};
use marian_model::ModelManifest;
use marian_tokenizer::Tokenizer;
use serde_json::json;
use sha2::{Digest, Sha256};

fn model_dir() -> PathBuf {
    std::env::var_os("MARIAN_CPU_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/enzh"))
}

#[test]
#[ignore = "requires MARIAN_Q8_MODEL pointing to a Marian Q8 binary v1 artifact"]
fn matches_known_token_translation() {
    let model_dir = model_dir();
    let manifest = ModelManifest::load(&model_dir).unwrap();
    let q8_path = std::env::var_os("MARIAN_Q8_MODEL")
        .map(PathBuf::from)
        .expect("MARIAN_Q8_MODEL");
    let shortlist = manifest.shortlist.as_ref().map(|path| model_dir.join(path));
    let engine = Q8CpuEngine::load(q8_path, shortlist.as_deref(), &manifest.architecture).unwrap();
    let output = engine
        .translate_token_ids(&[277, 2_904, 272, 1_800, 686, 265, 0], &[0, 7], &[512])
        .unwrap();
    assert_eq!(output.offsets, [0, 6]);
    assert_eq!(output.tokens, [2_068, 3_242, 626, 1_259, 265, 0]);

    let source = Tokenizer::open(model_dir.join(&manifest.source_vocab)).unwrap();
    let target = Tokenizer::open(model_dir.join(&manifest.target_vocab)).unwrap();
    let cases = [
        ("Hello, world!", "你好,世界!"),
        ("The weather is beautiful today.", "今天天气很美。"),
        ("Please open the window.", "请打开窗户。"),
        ("Thank you for your help!", "感谢您的帮助!"),
        ("Where is the nearest train station?", "最近的火车站在哪里?"),
    ];
    let mut packed = Vec::new();
    let mut offsets = vec![0_u32];
    for (text, _) in cases {
        packed.extend(source.encode(text).unwrap());
        packed.push(manifest.architecture.eos_id);
        offsets.push(packed.len() as u32);
    }
    let translated = engine
        .translate_token_ids(&packed, &offsets, &vec![512; cases.len()])
        .unwrap();
    for (index, (_, expected)) in cases.iter().enumerate() {
        let start = translated.offsets[index] as usize;
        let end = translated.offsets[index + 1] as usize;
        let ids = translated.tokens[start..end]
            .iter()
            .copied()
            .filter(|&token| token != manifest.architecture.eos_id)
            .collect::<Vec<_>>();
        assert_eq!(target.decode(&ids).unwrap(), *expected);
    }

    for (index, (source_text, expected)) in cases.iter().take(2).enumerate() {
        let mut single_tokens = source.encode(source_text).unwrap();
        single_tokens.push(manifest.architecture.eos_id);
        let single = engine
            .translate_token_ids(&single_tokens, &[0, single_tokens.len() as u32], &[512])
            .unwrap();
        let batch_start = translated.offsets[index] as usize;
        let batch_end = translated.offsets[index + 1] as usize;
        assert_eq!(single.tokens, translated.tokens[batch_start..batch_end]);
        let ids = single
            .tokens
            .iter()
            .copied()
            .filter(|&token| token != manifest.architecture.eos_id)
            .collect::<Vec<_>>();
        assert_eq!(target.decode(&ids).unwrap(), *expected);
    }
}

#[test]
#[ignore = "requires MARIAN_Q8_MODEL pointing to a Marian Q8 binary v1 artifact"]
fn verified_q8_backend_translates_text() {
    let source_dir = model_dir();
    let fp32_manifest = ModelManifest::load(&source_dir).unwrap();
    let q8_source = std::env::var_os("MARIAN_Q8_MODEL")
        .map(PathBuf::from)
        .expect("MARIAN_Q8_MODEL");
    let unique = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "marian-rust-q8-backend-{}-{unique}",
        std::process::id()
    ));
    fs::create_dir(&directory).unwrap();

    let weights_name = "model.intgemm.alphas.bin";
    link_or_copy(&q8_source, &directory.join(weights_name));
    for name in [
        &fp32_manifest.source_vocab,
        &fp32_manifest.target_vocab,
        fp32_manifest.shortlist.as_ref().unwrap(),
    ] {
        link_or_copy(&source_dir.join(name), &directory.join(name));
    }
    let manifest = json!({
        "format": fp32_manifest.format,
        "model_id": fp32_manifest.model_id,
        "source_lang": fp32_manifest.source_lang,
        "target_lang": fp32_manifest.target_lang,
        "weights": weights_name,
        "source_vocab": fp32_manifest.source_vocab,
        "target_vocab": fp32_manifest.target_vocab,
        "shortlist": fp32_manifest.shortlist,
        "precision": "q8",
        "architecture": {
            "model_dim": fp32_manifest.architecture.model_dim,
            "attention_heads": fp32_manifest.architecture.attention_heads,
            "encoder_layers": fp32_manifest.architecture.encoder_layers,
            "decoder_layers": fp32_manifest.architecture.decoder_layers,
            "ffn_dim": fp32_manifest.architecture.ffn_dim,
            "source_vocab_size": fp32_manifest.architecture.source_vocab_size,
            "target_vocab_size": fp32_manifest.architecture.target_vocab_size,
            "eos_id": fp32_manifest.architecture.eos_id,
            "unk_id": fp32_manifest.architecture.unk_id,
            "max_length_factor": 2,
        },
        "checksums": {
            "weights_sha256": sha256(&q8_source),
            "source_vocab_sha256": fp32_manifest.checksums.source_vocab_sha256,
            "target_vocab_sha256": fp32_manifest.checksums.target_vocab_sha256,
            "shortlist_sha256": fp32_manifest.checksums.shortlist_sha256,
        }
    });
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let mut backend = Q8CpuBackend::load(&directory).unwrap();
    assert_eq!(backend.info().precision, "q8");
    assert_eq!(backend.info().name, "cpu");
    let inputs = [
        TranslationInput::new("The weather is beautiful today.", "en", "zh"),
        TranslationInput::new("Please open the window.", "en", "zh"),
    ];
    let output = backend.translate_batch(&inputs).unwrap();
    assert_eq!(output[0].text, "今天天气很美。");
    assert_eq!(output[1].text, "请打开窗户。");

    let long_source = vec!["The weather is beautiful today."; 80].join(" ");
    let long_expected = vec!["今天天气很美。"; 80].join(" ");
    let paragraph_inputs = [
        TranslationInput::new(
            "First sentence.\nSecond sentence?\r\nThird sentence!",
            "en",
            "zh",
        ),
        TranslationInput::new(long_source, "en", "zh"),
    ];
    let paragraphs = backend.translate_batch(&paragraph_inputs).unwrap();
    assert_eq!(paragraphs[0].text, "第一句话。\n第二句话?\r\n第三句话!");
    assert_eq!(paragraphs[1].text, long_expected);
    assert!(paragraphs[1].input_tokens > 256);
    assert!(paragraphs[1].output_tokens <= paragraph_inputs[1].max_output_tokens);

    let mut limited = TranslationInput::new(
        "The weather is beautiful today. The weather is beautiful today.",
        "en",
        "zh",
    );
    limited.max_output_tokens = 1;
    let limited = backend.translate_batch(&[limited]).unwrap().remove(0);
    assert_eq!(limited.output_tokens, 1);
    assert_eq!(limited.text.trim(), "今天");
    assert!(limited.text.ends_with(' '));
    drop(backend);
    fs::remove_dir_all(directory).unwrap();
}

fn link_or_copy(source: &Path, destination: &Path) {
    if fs::hard_link(source, destination).is_err() {
        fs::copy(source, destination).unwrap();
    }
}

fn sha256(path: &Path) -> String {
    let mut file = fs::File::open(path).unwrap();
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).unwrap();
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    format!("{:x}", hasher.finalize())
}
