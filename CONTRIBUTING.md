# Contributing

Thanks for improving Marian MLX. Small, focused pull requests with a clear
failure case or benchmark are easiest to review.

## Before opening a pull request

1. Open an issue for a new backend, model format, or public API change.
2. Keep generated models, weights, build directories, and benchmark dumps out
   of Git. They are intentionally ignored.
3. Add tests for behavior changes and document user-visible flags or endpoints.
4. Run `make check` on every change.

The portable checks do not need Metal or a model. On Apple Silicon, changes to
the native backend should additionally run:

```sh
scripts/prepare-enzh-model.sh
scripts/build-release.sh
cargo test -p marian-tokenizer --test mozilla_enzh -- --ignored
cargo test -p marian-mlx --features metal --test golden -- --ignored
cargo test -p marian-cpu --release --test golden -- --ignored
cargo test -p marian-cpu --release --test q8_golden -- --ignored
```

The legacy `mlx` feature is only a compatibility alias. New commands, tests,
and documentation must use `metal`. The MSL source is embedded and compiled at
runtime, so native changes do not require an MLX submodule, `libmlx.dylib`, or
an external `.metallib`.

The ignored golden test is a smoke test, not a corpus-level quality result.
Performance changes should include the exact machine, precision, request
corpus, warmup, concurrency, and command. Do not claim GPU execution without a
Metal trace or equivalent device evidence.

## Code conventions

- Rust 1.86 is the minimum supported version and is pinned in
  `rust-toolchain.toml`.
- `cargo fmt` is authoritative for Rust formatting.
- Clippy warnings are treated as errors in CI.
- The GPU object stays on its dedicated owner thread. Do not add unsafe
  `Send`/`Sync` implementations to bypass that ownership model.
- Keep model ownership separate from kernel parallelism: the CPU model has one
  owner, while the configured 1/2/4 compute threads may participate in both
  FP32 matrix multiplication and Q8 row-parallel kernels.
- Keep the runtime boundary explicit: `marian-tokenizer`, `marian-cpu`, the
  long-text segmenter, and the Metal host are Rust; `objc2-metal` calls the
  system Metal framework on macOS.
- Q8 changes must retain strict tensor-set, shape, quantization, shortlist, and
  golden checks. Quality comparisons must report the corpus and exact-match
  count instead of claiming general equivalence from a small smoke test.
- Segmenter changes must test tokenizer piece limits, punctuation-free input,
  abbreviations, Unicode, whitespace/newline preservation, and output order.
- Production backends fail closed. The echo backend is only for API tests and
  must never be selected as an implicit fallback.
- Preserve bounded queues and explicit overload responses.

## Licensing

Contributions are accepted under the repository's MIT license. New external
code, model artifacts, or native libraries must have a compatible license and
their source revision recorded in `THIRD_PARTY_NOTICES.md`.

By submitting a contribution, you certify that you have the right to license
it under these terms.
