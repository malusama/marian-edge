# Third-party notices

The service code in this repository is MIT licensed. The projects and model
artifacts below retain their own terms.

## MLX

- Project: [ml-explore/mlx](https://github.com/ml-explore/mlx)
- Pinned revision: `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`
- License: MIT, copyright Apple Inc.
- Use: native Apple Silicon tensor and Metal runtime

MLX is a Git submodule. Native release archives include its license next to
the executable.

## Bergamot Translator

- Project: [browsermt/bergamot-translator](https://github.com/browsermt/bergamot-translator)
- License: Mozilla Public License 2.0
- Use: optional Linux CPU backend in the `:cpu` container image

The container build obtains a pinned source revision directly from the
upstream repository. The final image includes the MPL-2.0 license and the
corresponding source revision/URL.

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
and vulnerability policy checks before releases.
