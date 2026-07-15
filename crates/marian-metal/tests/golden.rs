#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::path::PathBuf;

use marian_core::{TranslationBackend, TranslationInput};
use marian_metal::MetalBackend;

#[test]
#[ignore = "requires converted Mozilla en-zh weights and an Apple GPU"]
fn matches_known_sentences() {
    let model_dir = std::env::var_os("MARIAN_EDGE_MODEL_DIR")
        .or_else(|| std::env::var_os("MARIAN_MLX_MODEL_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../../models/enzh"));
    let mut backend = MetalBackend::load(model_dir).unwrap();
    let cases = [
        ("Hello, world!", "你好,世界!"),
        ("The weather is beautiful today.", "今天天气很美。"),
        ("Please open the window.", "请打开窗户。"),
        ("Thank you for your help!", "感谢您的帮助!"),
        ("Where is the nearest train station?", "最近的火车站在哪里?"),
    ];
    let inputs: Vec<_> = cases
        .iter()
        .map(|(source, _)| TranslationInput::new(*source, "en", "zh"))
        .collect();
    let outputs = backend.translate_batch(&inputs).unwrap();
    for ((_, expected), actual) in cases.iter().zip(outputs) {
        assert_eq!(&actual.text, expected);
    }

    // Immersive Translate reserves numbered and paired-tag placeholders. This
    // is a protocol-compatibility regression, separate from the five release
    // translation goldens above.
    let placeholder_cases = [
        ("{0} Hello {1} world.", "{0} Hello {1} 世界。"),
        (
            "<b0></b0> Hello <b1></b1> world.",
            "<b0></b0> Hello <b1></b1> 世界。",
        ),
    ];
    let placeholder_inputs = placeholder_cases
        .iter()
        .map(|(source, _)| TranslationInput::new(*source, "en", "zh"))
        .collect::<Vec<_>>();
    let placeholder_outputs = backend.translate_batch(&placeholder_inputs).unwrap();
    for ((_, expected), actual) in placeholder_cases.iter().zip(placeholder_outputs) {
        assert_eq!(&actual.text, expected);
    }

    // The old MLX backend built one shortlist union for the entire packed
    // batch, so neighboring requests could change a sentence's wording. The
    // direct Metal decoder keeps candidates and length limits per row.
    let invariant_cases = [
        "Rust is a systems programming language.",
        "Quantum entanglement puzzled Einstein.",
    ];
    let invariant_inputs = invariant_cases
        .iter()
        .map(|source| TranslationInput::new(*source, "en", "zh"))
        .collect::<Vec<_>>();
    let batched = backend.translate_batch(&invariant_inputs).unwrap();
    for (input, batched_output) in invariant_inputs.iter().zip(batched) {
        let single = backend
            .translate_batch(std::slice::from_ref(input))
            .unwrap();
        assert_eq!(single[0].text, batched_output.text);
    }

    let paragraph_inputs = [
        TranslationInput::new(
            "First sentence.\nSecond sentence?\r\nThird sentence!",
            "en",
            "zh",
        ),
        TranslationInput::new(
            vec!["The weather is beautiful today."; 80].join(" "),
            "en",
            "zh",
        ),
    ];
    let paragraphs = backend.translate_batch(&paragraph_inputs).unwrap();
    assert_eq!(paragraphs[0].text, "第一句话。\n第二句话?\r\n第三句话!");
    assert_eq!(paragraphs[1].text, vec!["今天天气很美。"; 80].join(" "));

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
}
