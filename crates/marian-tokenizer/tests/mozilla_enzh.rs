use std::path::PathBuf;

use marian_tokenizer::Tokenizer;

// Token ids and normalized text below were captured from the native Google
// SentencePiece 0.2.1 implementation before removing it from production.

fn model_dir() -> PathBuf {
    std::env::var_os("MARIAN_TOKENIZER_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/enzh"))
}

fn source() -> Tokenizer {
    Tokenizer::open(model_dir().join("source.spm")).expect("load real Mozilla source model")
}

fn target() -> Tokenizer {
    Tokenizer::open(model_dir().join("target.spm")).expect("load real Mozilla target model")
}

#[test]
#[ignore = "requires the real Mozilla en-zh source.spm model"]
fn source_model_matches_cpp_golden() {
    let tokenizer = source();
    assert_eq!(tokenizer.len(), 32_000);
    assert_eq!(
        tokenizer.encode("Hello, world!").unwrap(),
        [15_090, 264, 357, 470]
    );
    assert_eq!(
        tokenizer.encode("The weather is beautiful today.").unwrap(),
        [277, 2_904, 272, 1_800, 686, 265]
    );
}

#[test]
#[ignore = "requires the real Mozilla en-zh target.spm model"]
fn target_model_matches_cpp_golden() {
    let tokenizer = target();
    assert_eq!(tokenizer.len(), 32_000);
    assert_eq!(
        tokenizer.encode("今天天气很美。").unwrap(),
        [2_068, 3_242, 626, 1_259, 265]
    );
    assert_eq!(
        tokenizer.decode(&[2_068, 3_242, 626, 1_259, 265]).unwrap(),
        "今天天气很美。"
    );
}

#[test]
#[ignore = "requires the real Mozilla en-zh source.spm and target.spm models"]
fn unicode_normalization_and_byte_fallback_match_cpp_golden() {
    let text = "Ｆｕｌｌ－ｗｉｄｔｈ　café\t你好 👩‍💻";

    let source = source();
    let source_ids = source.encode(text).unwrap();
    assert_eq!(
        source_ids,
        [
            6_654, 275, 29_679, 16_143, 202, 176, 278, 235, 196, 167, 236, 172, 196, 278, 247, 166,
            152, 176, 278, 247, 166, 153, 194,
        ]
    );
    assert_eq!(
        source.decode(&source_ids).unwrap(),
        "Full-width café 你好 👩 💻"
    );

    let target = target();
    let target_ids = target.encode(text).unwrap();
    assert_eq!(
        target_ids,
        [
            1_318, 8_724, 320, 2_712, 2_598, 2_340, 7_438, 9_188, 5_011, 679, 521, 264, 247, 166,
            152, 176, 264, 247, 166, 153, 194,
        ]
    );
    assert_eq!(
        target.decode(&target_ids).unwrap(),
        "Full-width café 你好 👩 💻"
    );
}

#[test]
#[ignore = "requires the real Mozilla en-zh source.spm and target.spm models"]
fn user_defined_symbols_remain_atomic() {
    let source = source();
    assert_eq!(source.encode("__source__").unwrap(), [278, 2]);
    assert_eq!(
        source.encode("before __target__ after").unwrap(),
        [478, 278, 3, 439]
    );

    let target = target();
    assert_eq!(target.encode("__source__").unwrap(), [264, 2]);
    assert_eq!(
        target.encode("before __target__ after").unwrap(),
        [11_730, 12_994, 593, 264, 3, 264, 9_188, 2_904]
    );
}

#[test]
#[ignore = "requires the real Mozilla en-zh source.spm and target.spm models"]
fn encode_decode_applies_model_normalization() {
    for tokenizer in [source(), target()] {
        for (text, normalized) in [
            ("  Hello   world  ", "Hello world"),
            ("e\u{301} Å ﬃ", "é Å ffi"),
            ("🦀🚀\u{200d}✨", "🦀🚀 ✨"),
        ] {
            let ids = tokenizer.encode(text).unwrap();
            assert_eq!(tokenizer.decode(&ids).unwrap(), normalized);
        }
    }
}
