#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::path::PathBuf;

use marian_core::{TranslationBackend, TranslationInput};
use marian_cpu::CpuBackend;
use marian_metal::MetalBackend;

const PRODUCTION_BATCH_SIZE: usize = 16;

fn model_dir() -> PathBuf {
    std::env::var_os("MARIAN_EDGE_MODEL_DIR")
        .or_else(|| std::env::var_os("MARIAN_MLX_MODEL_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/enzh"))
}

fn deterministic_corpus() -> Vec<String> {
    const ADJECTIVES: [&str; 8] = [
        "small",
        "large",
        "quiet",
        "bright",
        "ancient",
        "modern",
        "careful",
        "unexpected",
    ];
    const NOUNS: [&str; 8] = [
        "train",
        "window",
        "compiler",
        "teacher",
        "river",
        "market",
        "satellite",
        "library",
    ];
    const VERBS: [&str; 8] = [
        "crosses", "opens", "tests", "observes", "builds", "explains", "finds", "protects",
    ];
    const ENDINGS: [&str; 4] = [
        "today.",
        "at noon.",
        "without warning!",
        "near the station?",
    ];

    let mut corpus = Vec::with_capacity(200);
    'templates: for adjective in ADJECTIVES {
        for noun in NOUNS {
            for verb in VERBS {
                for ending in ENDINGS {
                    let index = corpus.len();
                    corpus.push(format!(
                        "{index}: The {adjective} {noun} {verb} the old bridge {ending}"
                    ));
                    if corpus.len() == 180 {
                        break 'templates;
                    }
                }
            }
        }
    }

    corpus.extend(
        [
            "Hello, world!",
            "Numbers: 0, 1, 2, 3.14159, and 2026-07-15.",
            "Quotes “like this”, apostrophes, em—dashes, and ellipses… should work.",
            "Café naïve résumé coöperate — Unicode normalization matters.",
            "Emoji test: 🚄🌧️🧪; please keep translating the surrounding English.",
            "Line one\nLine two\tTabbed text.",
            "UPPERCASE and lowercase MixedCase words.",
            "A",
            "I do not know.",
            "Where is platform number twelve?",
            "Rust prevents many memory errors, but careful design still matters.",
            "If the weather changes tomorrow, please close every open window.",
            "The quick brown fox jumps over the lazy dog.",
            "Zero-width? abc\u{200b}def and non-breaking\u{a0}space.",
            "Repeated punctuation!!!???...",
            "The CPU batch must match each sentence translated alone.",
            "A longer sentence with several clauses, commas, numbers 42 and 9000, plus a final question: does it still agree?",
            "Thank you for your help!",
            "Please open the window.",
            "The weather is beautiful today.",
        ]
        .into_iter()
        .map(str::to_owned),
    );

    assert_eq!(corpus.len(), 200);
    corpus
}

fn translate_production_batches<B: TranslationBackend>(
    backend: &mut B,
    inputs: &[TranslationInput],
) -> Vec<marian_core::TranslationOutput> {
    inputs
        .chunks(PRODUCTION_BATCH_SIZE)
        .flat_map(|batch| backend.translate_batch(batch).unwrap())
        .collect()
}

#[test]
#[ignore = "requires converted Mozilla en-zh weights and an Apple GPU; set MARIAN_EDGE_MODEL_DIR to override the model directory"]
fn cpu_fp32_matches_direct_metal_for_deterministic_corpus() {
    let corpus = deterministic_corpus();
    let shortest = corpus.iter().map(String::len).min().unwrap();
    let longest = corpus.iter().map(String::len).max().unwrap();
    assert!(longest > shortest * 8, "corpus must exercise mixed padding");
    assert!(corpus.chunks(PRODUCTION_BATCH_SIZE).any(|batch| {
        let shortest = batch.iter().map(String::len).min().unwrap();
        let longest = batch.iter().map(String::len).max().unwrap();
        shortest != longest
    }));

    let inputs = corpus
        .iter()
        .map(|text| TranslationInput::new(text, "en", "zh"))
        .collect::<Vec<_>>();
    let model_dir = model_dir();
    let mut cpu = CpuBackend::load(&model_dir).unwrap();
    let mut metal = MetalBackend::load(&model_dir).unwrap();
    let mixed_f16 = metal.info().precision == "mixed-f16";

    let cpu_batch = translate_production_batches(&mut cpu, &inputs);
    let metal_batch = translate_production_batches(&mut metal, &inputs);
    assert_eq!(cpu_batch.len(), corpus.len());
    assert_eq!(metal_batch.len(), corpus.len());
    let mut mismatches = Vec::new();
    for (index, ((source, cpu_output), metal_output)) in
        corpus.iter().zip(&cpu_batch).zip(&metal_batch).enumerate()
    {
        if cpu_output != metal_output {
            if mixed_f16 {
                mismatches.push(index);
            } else {
                assert_eq!(
                    cpu_output, metal_output,
                    "CPU and Metal differ for corpus item {index}: {source:?}"
                );
            }
        }
    }
    if mixed_f16 {
        let exact = corpus.len() - mismatches.len();
        eprintln!(
            "mixed-f16 exact translations: {exact}/{}; mismatches: {mismatches:?}",
            corpus.len()
        );
        assert!(
            exact >= 198,
            "mixed-f16 must retain at least 99% exact translations on the deterministic corpus"
        );
    }

    // Sampling across templates and every special-input region keeps the test
    // bounded while proving that padding and neighboring shortlist rows cannot
    // change either backend's result.
    for index in [0, 1, 17, 63, 127, 179, 180, 183, 184, 193, 199] {
        let input = std::slice::from_ref(&inputs[index]);
        let cpu_single = cpu.translate_batch(input).unwrap();
        let metal_single = metal.translate_batch(input).unwrap();
        assert_eq!(
            cpu_single[0], cpu_batch[index],
            "CPU batch drift at {index}"
        );
        assert_eq!(
            metal_single[0], metal_batch[index],
            "Metal batch drift at {index}"
        );
        if !mixed_f16 {
            assert_eq!(
                cpu_single[0], metal_single[0],
                "CPU and Metal single-item drift at {index}"
            );
        }
    }
}
