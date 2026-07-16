# Third-party notices

The service code in this repository is MIT licensed. The projects and model
artifacts below retain their own terms.

## Apple Metal

The native Apple Silicon backend calls the Metal framework supplied by macOS
directly and compiles the repository's own Metal kernels at runtime. Native
release archives do not redistribute MLX, a third-party dynamic library, or a
precompiled Metal library.

## SentencePiece

- Project: [VoiceLessQ/sentencepiece-rust](https://github.com/VoiceLessQ/sentencepiece-rust)
- Version: 0.1.1
- License: Apache License 2.0
- Use: source and target model tokenization

`marian-tokenizer` uses this pure-Rust SentencePiece implementation. It reads
the existing `.spm` model format without linking or
redistributing the native Google SentencePiece library. A verbatim copy of the
crate's license is distributed as [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0).

## Mozilla translation model artifacts

- Registry: [Mozilla translation model registry](https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/db/models.json)
- Archived tooling repository:
  [mozilla/firefox-translations-models](https://github.com/mozilla/firefox-translations-models)
- Direction used by the current release: English to Chinese `base-memory`

Model files are not committed, attached to releases, or embedded in container
images. The install/model-preparation scripts download them directly from the
Mozilla-operated registry at the operator's request, verify pinned SHA-256
digests, and create local runtime files. The registry record itself does not
declare a per-model license; operators must review the upstream terms before
use or redistribution.

Firefox and Mozilla are trademarks of the Mozilla Foundation in the United
States and other countries.

## Rust crates

Rust dependencies and exact versions are recorded in `Cargo.lock`. Each crate
continues to be governed by the license declared by that crate. CI runs license
and vulnerability policy checks before releases. This includes the
`objc2-metal` bindings used to call Apple's system Metal framework directly,
`matrixmultiply` used by the pure-Rust FP32 CPU backend, `rten-gemm` 0.21.0 for
the pure-Rust Q8 linear kernels, and the pure-Rust `sentencepiece-rust`
inference crate. The long-text segmenter is repository-owned Rust code. MLX,
Google SentencePiece, and a native inference library are not runtime
dependencies.
