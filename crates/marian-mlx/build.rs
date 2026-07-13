fn main() {
    #[cfg(feature = "mlx")]
    build_mlx_bridge();
}

#[cfg(feature = "mlx")]
fn build_mlx_bridge() {
    use std::{env, path::PathBuf};

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64")
    {
        panic!("the MLX backend requires Apple Silicon macOS");
    }

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let default_prefix = manifest_dir.join("../../build/mlx-install");
    let prefix = env::var_os("MLX_PREFIX")
        .map(PathBuf::from)
        .unwrap_or(default_prefix);
    let header = prefix.join("include/mlx/mlx.h");
    if !header.exists() {
        panic!(
            "MLX was not found at {}. Run scripts/build-mlx.sh or set MLX_PREFIX.",
            prefix.display()
        );
    }

    cxx_build::bridge("src/ffi.rs")
        .file("native/engine.cpp")
        .include("native/include")
        .include(prefix.join("include"))
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-O3")
        .flag_if_supported("-DNDEBUG")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated-copy")
        .compile("marian_mlx_bridge");

    println!(
        "cargo:rustc-link-search=native={}",
        prefix.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=mlx");
    // Release bundles place libmlx.dylib beside the server executable.
    // A relative rpath keeps the binary portable across installation paths.
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=native/engine.cpp");
    println!("cargo:rerun-if-changed=native/include/engine.hpp");
    println!("cargo:rerun-if-env-changed=MLX_PREFIX");
}
