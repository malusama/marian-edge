use std::{env, path::PathBuf, process::Command};

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest_dir.join("../..");
    emit_git_rerun_paths(&workspace);
    let revision = env::var("MARIAN_EDGE_BUILD_GIT_SHA")
        .or_else(|_| env::var("MARIAN_MLX_BUILD_GIT_SHA"))
        .ok()
        .or_else(|| {
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
        "cargo:rustc-env=MARIAN_EDGE_BUILD_GIT_SHA={}",
        revision
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or("unknown")
    );
    println!("cargo:rerun-if-env-changed=MARIAN_EDGE_BUILD_GIT_SHA");
    println!("cargo:rerun-if-env-changed=MARIAN_MLX_BUILD_GIT_SHA");
}

fn emit_git_rerun_paths(workspace: &std::path::Path) {
    println!(
        "cargo:rerun-if-changed={}",
        workspace.join(".git/HEAD").display()
    );

    let Ok(output) = Command::new("git")
        .args(["symbolic-ref", "-q", "HEAD"])
        .current_dir(workspace)
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }

    let reference = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let Ok(path_output) = Command::new("git")
        .args(["rev-parse", "--git-path", &reference])
        .current_dir(workspace)
        .output()
    else {
        return;
    };
    if path_output.status.success() {
        let path = PathBuf::from(String::from_utf8_lossy(&path_output.stdout).trim());
        let path = if path.is_absolute() {
            path
        } else {
            workspace.join(path)
        };
        println!("cargo:rerun-if-changed={}", path.display());
    }
}
