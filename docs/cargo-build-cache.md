# Local Cargo build cache

Local development supports [huacnlee/cargo v0.102.0-4](https://github.com/huacnlee/cargo/releases/tag/v0.102.0-4).
It shares Rust intermediate files across Git worktrees and matching dependency
builds across projects. The repository's Rust compiler version, targets, features
and release optimization settings remain authoritative.

## Setup and rollback

Install the pinned fork in its own directory, then select it for the current shell:

```sh
curl -fsSL https://raw.githubusercontent.com/huacnlee/cargo/v0.102.0-4/install.sh | \
  sh -s -- --version 0.102.0-4 --dir "$HOME/.local/share/huacnlee-cargo/0.102.0-4"
export PATH="$HOME/.local/share/huacnlee-cargo/0.102.0-4:$PATH"
cargo -V
```

The version must include `huacnlee/cargo 0.102.0-4`. Remove that PATH entry to
return to the previous Cargo; this installation does not replace rustup's Cargo.
No binary, home-directory path or machine-specific launcher is tracked here.

## Cache policy and output paths

`Cargo.toml` explicitly retains Mach-O symbols in dev (and inherited test)
and release profiles. With Rust 1.86 on macOS 27, stripping proc-macro dylibs
can produce `mis-aligned LINKEDIT string pool` load failures. Keeping this
setting in the workspace makes direct fork builds work without a local launcher
override; it does not change the compiler pin or model execution settings.

`.cargo/config.toml` limits the fork's shared dependency cache to **10 GiB**,
expires unused units after one day and runs collection at most once per hour.
This cap applies to shared dependencies, not the total of all project caches.
Project caches use the fork's automatic size limit. Final build outputs remain
local to each worktree so one branch cannot link another branch's archive.

Upstream Cargo remains supported and may warn that `build.gc` is unused; CI's
existing compiler caches and toolchain selection continue to work. Keep the
existing `RUSTC_WRAPPER` when using sccache, and do not stack it with another
compiler wrapper.

This fork ignores `CARGO_TARGET_DIR`, `--target-dir` and `build.target-dir`.
For a build that requires a particular output directory, use the upstream Cargo
from the active Rust toolchain:

```sh
rustup run "$(rustup show active-toolchain | cut -d ' ' -f 1)" cargo build --locked --target-dir /path/to/build-output
```

Older target directories are not migrated automatically. Review unused compiler
intermediates separately; preserve app bundles, packaged libraries, source files,
saves and outputs held open by running builds. Installing the tool alone does not
establish a speedup or reclaim existing disk usage.
