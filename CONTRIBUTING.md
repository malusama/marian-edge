# Contributing

Thanks for improving Marian MLX. Small, focused pull requests with a clear
failure case or benchmark are easiest to review.

## Before opening a pull request

1. Open an issue for a new backend, model format, or public API change.
2. Keep generated models, weights, build directories, and benchmark dumps out
   of Git. They are intentionally ignored.
3. Add tests for behavior changes and document user-visible flags or endpoints.
4. Run `make check` on every change.

The portable checks do not need MLX or a model. On Apple Silicon, changes to
the native backend should additionally run:

```sh
git submodule update --init --recursive
scripts/build-mlx.sh
scripts/prepare-enzh-model.sh
scripts/build-release.sh
cargo test -p marian-mlx --features mlx --test golden -- --ignored
```

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
- Production backends fail closed. The echo backend is only for API tests and
  must never be selected as an implicit fallback.
- Preserve bounded queues and explicit overload responses.

## Licensing

Contributions are accepted under the repository's MIT license. New external
code, model artifacts, or native libraries must have a compatible license and
their source revision recorded in `THIRD_PARTY_NOTICES.md`.

By submitting a contribution, you certify that you have the right to license
it under these terms.
