#![cfg(feature = "mlx")]

use std::path::PathBuf;

use marian_core::{TranslationBackend, TranslationInput};
use marian_mlx::MlxBackend;

#[test]
#[ignore = "requires converted Mozilla en-zh weights and an Apple GPU"]
fn matches_known_bergamot_sentences() {
    let model_dir = std::env::var_os("MARIAN_MLX_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../../models/enzh"));
    let mut backend = MlxBackend::load(model_dir).unwrap();
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
}
