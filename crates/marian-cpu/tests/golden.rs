use std::path::PathBuf;

use marian_core::{TranslationBackend, TranslationInput};
use marian_cpu::CpuBackend;

fn model_dir() -> PathBuf {
    std::env::var_os("MARIAN_CPU_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/enzh"))
}

#[test]
#[ignore = "requires converted Mozilla en-zh weights; FP32 CPU inference is intentionally slow"]
fn matches_known_sentences_and_batch_semantics() {
    let mut backend = CpuBackend::load(model_dir()).unwrap();
    let token_output = backend
        .engine()
        .translate_token_ids(&[277, 2_904, 272, 1_800, 686, 265, 0], &[0, 7], &[512])
        .unwrap();
    assert_eq!(token_output.offsets, [0, 6]);
    assert_eq!(token_output.tokens, [2_068, 3_242, 626, 1_259, 265, 0]);

    let cases = [
        ("Hello, world!", "你好,世界!"),
        ("The weather is beautiful today.", "今天天气很美。"),
        ("Please open the window.", "请打开窗户。"),
        ("Thank you for your help!", "感谢您的帮助!"),
        ("Where is the nearest train station?", "最近的火车站在哪里?"),
    ];
    let inputs = cases
        .iter()
        .map(|(source, _)| TranslationInput::new(*source, "en", "zh"))
        .collect::<Vec<_>>();
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
}
