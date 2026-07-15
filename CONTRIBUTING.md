# Contributing

Thanks for improving Marian MLX. Small, focused pull requests with a clear
failure case or benchmark are easiest to review.

## Before opening a pull request

1. Open an issue for a new backend, model format, or public API change.
2. Keep generated models, weights, build directories, and benchmark dumps out
   of Git. They are intentionally ignored.
3. Add tests for behavior changes and document user-visible flags or endpoints.
4. Run `make check` on every change; it includes documentation contracts.

The portable checks do not need Metal or a model. On Apple Silicon, changes to
the native backend should additionally run:

```sh
scripts/prepare-enzh-model.sh
scripts/build-release.sh
cargo test -p marian-tokenizer --test mozilla_enzh -- --ignored
cargo test -p marian-metal --features metal --test golden -- --ignored
MARIAN_MLX_MODEL_DIR=models/enzh cargo test -p marian-metal --features metal \
  --release --test cpu_metal_differential -- --ignored
MARIAN_CPU_MODEL_DIR="$PWD/models/enzh" \
  cargo test -p marian-cpu --release --test golden -- --ignored
MARIAN_Q8_MODEL=/absolute/path/to/model.intgemm.alphas.bin \
MARIAN_CPU_MODEL_DIR="$PWD/models/enzh" \
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

## Release checklist

Release tags drive both the macOS archive and the multi-architecture CPU image.
Complete the following on the release commit before creating a tag:

1. Update the workspace and repository-owned dependency versions in
   `Cargo.toml`, regenerate `Cargo.lock`, and update the Dockerfile's default
   version. The `vX.Y.Z` tag must exactly match the workspace version.
2. Move the relevant `CHANGELOG.md` entries out of `Unreleased`, add the release
   date, update its comparison links, and switch both READMEs' pinned installer
   and container examples to the exact version about to be tagged. The tag will
   freeze this same commit, so these changes cannot be deferred until after it.
3. Run `make check`, `scripts/package-macos.sh`, the Metal profiler parser
   tests, and the ignored
   model-backed golden/differential tests listed above. Verify the packaged
   archive and its checksums, not only the build tree.
4. Let CI pass on the exact release commit, then create and push the immutable
   `vX.Y.Z` tag. Never move a published tag or replace an existing asset.

After pushing the tag:

5. Confirm the release workflow publishes
   `marian-mlx-macos-arm64.tar.gz`, `SHA256SUMS`, and `install-macos.sh`, and
   that their provenance attestations are present.
6. Confirm `ghcr.io/malusama/marian-mlx:cpu-X.Y.Z` and the floating `:cpu` tag
   both contain Linux AMD64 and ARM64 manifests built from the tagged commit.
7. Exercise a fresh pinned macOS install, `/readyz`, `/info`, update, rollback,
   and uninstall. Confirm the backend, revision, device, precision, and model
   reported by `/info` match the intended release. If a published artifact is
   wrong, fix it in a new patch release instead of changing the existing tag.

## Licensing

Contributions are accepted under the repository's MIT license. New external
code, model artifacts, or native libraries must have a compatible license and
their source revision recorded in `THIRD_PARTY_NOTICES.md`.

By submitting a contribution, you certify that you have the right to license
it under these terms.
