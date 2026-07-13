use std::{env, path::PathBuf, process::Command};

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest_dir.join("../..");
    let revision = env::var("MARIAN_MLX_BUILD_GIT_SHA").ok().or_else(|| {
        let output = Command::new("git")
            .args(["rev-parse", "--short=12", "HEAD"])
            .current_dir(&workspace)
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    });
    println!(
        "cargo:rustc-env=MARIAN_MLX_BUILD_GIT_SHA={}",
        revision
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or("unknown")
    );
    println!("cargo:rerun-if-env-changed=MARIAN_MLX_BUILD_GIT_SHA");

    #[cfg(feature = "mlx")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
    }
}
