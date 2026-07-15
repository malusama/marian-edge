use std::{
    env,
    hint::black_box,
    path::PathBuf,
    time::{Duration, Instant},
};

use marian_cpu::{
    Q8Linear, Q8LinearScratch,
    benchmarking::{LayerNorm384, Shortlist384, attention_384},
    segment_text,
};
use marian_tokenizer::Tokenizer;
use serde_json::{Value, json};

fn samples<T>(iterations: usize, mut operation: impl FnMut() -> T) -> (Vec<u128>, T) {
    let mut timings = Vec::with_capacity(iterations);
    let mut last = None;
    for _ in 0..iterations {
        let started = Instant::now();
        last = Some(black_box(operation()));
        timings.push(started.elapsed().as_nanos());
    }
    (timings, last.expect("iterations must be positive"))
}

fn report(name: &str, mut values: Vec<u128>, metadata: Value) -> Value {
    values.sort_unstable();
    let percentile = |fraction: f64| {
        let index = ((values.len() as f64 * fraction).ceil() as usize)
            .saturating_sub(1)
            .min(values.len() - 1);
        values[index]
    };
    json!({
        "name": name,
        "iterations": values.len(),
        "p50_ns": percentile(0.50),
        "p95_ns": percentile(0.95),
        "p99_ns": percentile(0.99),
        "metadata": metadata,
    })
}

fn q8(iterations: usize, rows: usize) -> Value {
    let input_dim = 384;
    let output_dim = if rows == 1 { 384 } else { 1536 };
    let weights = (0..output_dim * input_dim)
        .map(|index| ((index * 31 % 127) as i16 - 63) as i8)
        .collect();
    let linear = Q8Linear::new("bench", input_dim, output_dim, weights, 32.0, 32.0, None)
        .expect("synthetic Q8 linear");
    let input = (0..rows * input_dim)
        .map(|index| ((index * 13 % 101) as f32 - 50.0) / 50.0)
        .collect::<Vec<_>>();
    let mut scratch = Q8LinearScratch::default();
    let mut destination = Vec::new();
    for _ in 0..10 {
        linear
            .run_into(black_box(&input), rows, &mut destination, &mut scratch)
            .unwrap();
        black_box(&destination);
    }
    let (timings, checksum) = samples(iterations, || {
        linear
            .run_into(black_box(&input), rows, &mut destination, &mut scratch)
            .unwrap();
        destination.first().copied().unwrap_or_default()
    });
    black_box(checksum);
    report(
        if rows == 1 {
            "q8_gemv_384"
        } else {
            "q8_gemm_16x384x1536"
        },
        timings,
        json!({"rows": rows, "input": input_dim, "output": output_dim, "kernel": linear.kernel_name(), "path": format!("{:?}", linear.execution_path())}),
    )
}

fn main() {
    let iterations = env::var("MARIAN_BENCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100)
        .max(1);
    let sequence = 32;
    let elements = sequence * 384;
    let a = (0..elements)
        .map(|index| (index % 97) as f32 / 97.0)
        .collect::<Vec<_>>();
    let b = (0..elements)
        .map(|index| (index % 89) as f32 / 89.0)
        .collect::<Vec<_>>();
    let c = (0..elements)
        .map(|index| (index % 83) as f32 / 83.0)
        .collect::<Vec<_>>();
    let decoder = &a[..384];
    let candidates = (0..2_048_u32).collect::<Vec<_>>();
    let normalization = LayerNorm384::new();
    let shortlist = Shortlist384::new(candidates.len()).unwrap();
    let long_text = vec!["The weather is beautiful today."; 80].join(" ");
    let mut output = vec![q8(iterations, 1), q8(iterations, 16)];

    let (timings, value) = samples(iterations, || {
        attention_384(black_box(&a), black_box(&b), black_box(&c), sequence).unwrap()
    });
    black_box(value);
    output.push(report(
        "attention_32x384",
        timings,
        json!({"sequence": sequence, "heads": 8}),
    ));

    let (timings, value) = samples(iterations, || {
        normalization
            .residual(black_box(&a), black_box(&b))
            .unwrap()
    });
    black_box(value);
    output.push(report(
        "residual_layer_norm_32x384",
        timings,
        json!({"rows": sequence}),
    ));

    let mut state = vec![0.0_f32; elements];
    let (timings, value) = samples(iterations, || {
        normalization
            .ssru(black_box(&a), black_box(&b), &mut state, black_box(&c))
            .unwrap()
    });
    black_box(value);
    output.push(report("ssru_32x384", timings, json!({"rows": sequence})));

    let (timings, value) = samples(iterations, || {
        shortlist
            .score(black_box(decoder), black_box(&candidates))
            .unwrap()
    });
    black_box(value);
    output.push(report(
        "shortlist_2048x384",
        timings,
        json!({"candidates": candidates.len()}),
    ));

    let (timings, value) = samples(iterations, || {
        segment_text(black_box(&long_text), 255, |text| {
            Ok(text.split_whitespace().count())
        })
        .unwrap()
    });
    black_box(value);
    output.push(report(
        "long_text_planning_80_sentences",
        timings,
        json!({"sentences": 80}),
    ));

    if let Some(model_dir) = env::var_os("MARIAN_CPU_MODEL_DIR").map(PathBuf::from) {
        let tokenizer = Tokenizer::open(model_dir.join("source.spm")).expect("source tokenizer");
        let (timings, value) = samples(iterations, || {
            tokenizer.encode(black_box(&long_text)).unwrap()
        });
        black_box(value);
        output.push(report(
            "tokenizer_long_text",
            timings,
            json!({"sentences": 80}),
        ));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "marian-mlx.microbenchmark.v1",
            "iterations": iterations,
            "results": output,
        }))
        .unwrap()
    );
    std::thread::sleep(Duration::from_millis(1));
}
