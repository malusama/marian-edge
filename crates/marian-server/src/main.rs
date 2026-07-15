use std::{net::SocketAddr, path::PathBuf, time::Duration};

use anyhow::{Context, Result};
use axum::http::HeaderValue;
use clap::{Parser, ValueEnum};
use marian_core::{EchoBackend, SchedulerConfig, Translator};
use marian_server::{AppState, router};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendKind {
    Auto,
    #[value(alias = "mlx")]
    Metal,
    #[value(alias = "cpu-fp32", alias = "cpu-q8", alias = "rust")]
    Cpu,
    Echo,
}

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    #[arg(long, env = "MARIAN_EDGE_BIND", default_value = "127.0.0.1:3000")]
    bind: SocketAddr,

    #[arg(long, env = "MARIAN_EDGE_BACKEND", value_enum, default_value = "auto")]
    backend: BackendKind,

    #[arg(long, env = "MARIAN_EDGE_MODEL_DIR", default_value = "models/enzh")]
    model_dir: PathBuf,

    #[arg(
        long,
        env = "MARIAN_EDGE_CPU_THREADS",
        default_value_t = 1,
        help = "Pure Rust CPU inference threads: 1, 2, or 4"
    )]
    cpu_threads: usize,

    #[arg(long, env = "MARIAN_EDGE_QUEUE_CAPACITY", default_value_t = 256)]
    queue_capacity: usize,

    #[arg(long, env = "MARIAN_EDGE_MAX_BATCH_SIZE", default_value_t = 16)]
    max_batch_size: usize,

    #[arg(
        long,
        env = "MARIAN_EDGE_MAX_PADDED_SOURCE_CHARS",
        default_value_t = 4096
    )]
    max_padded_source_chars: usize,

    #[arg(long, env = "MARIAN_EDGE_BATCH_WINDOW_US", default_value_t = 750)]
    batch_window_us: u64,

    #[arg(long, env = "MARIAN_EDGE_REQUEST_TIMEOUT_MS", default_value_t = 30_000)]
    request_timeout_ms: u64,

    #[arg(long, env = "MARIAN_EDGE_CORS_ORIGIN")]
    cors_origin: Option<HeaderValue>,

    #[arg(long, env = "MARIAN_EDGE_JSON_LOGS", default_value_t = false)]
    json_logs: bool,
}

fn main() -> Result<()> {
    apply_legacy_env_aliases()?;
    let args = Args::parse();
    configure_cpu_threads(&args)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to create Tokio runtime")?;
    runtime.block_on(run(args))
}

fn apply_legacy_env_aliases() -> Result<()> {
    for suffix in [
        "BIND",
        "BACKEND",
        "MODEL_DIR",
        "CPU_THREADS",
        "QUEUE_CAPACITY",
        "MAX_BATCH_SIZE",
        "MAX_PADDED_SOURCE_CHARS",
        "BATCH_WINDOW_US",
        "REQUEST_TIMEOUT_MS",
        "CORS_ORIGIN",
        "JSON_LOGS",
    ] {
        let primary_name = format!("MARIAN_EDGE_{suffix}");
        let legacy_name = format!("MARIAN_MLX_{suffix}");
        match (
            std::env::var_os(&primary_name),
            std::env::var_os(&legacy_name),
        ) {
            (Some(primary), Some(legacy)) if primary != legacy => anyhow::bail!(
                "conflicting settings: {primary_name} and legacy {legacy_name} have different values"
            ),
            (None, Some(legacy)) => {
                // SAFETY: this runs before Clap parsing, the Tokio runtime, or
                // any worker thread exists.
                unsafe { std::env::set_var(primary_name, legacy) };
            }
            _ => {}
        }
    }
    Ok(())
}

async fn run(args: Args) -> Result<()> {
    init_tracing(args.json_logs);
    let config = SchedulerConfig {
        queue_capacity: args.queue_capacity,
        max_batch_size: args.max_batch_size,
        max_padded_source_chars: args.max_padded_source_chars,
        batch_window: Duration::from_micros(args.batch_window_us),
        request_timeout: Duration::from_millis(args.request_timeout_ms),
        ..SchedulerConfig::default()
    };

    let translator = create_translator(args.backend, args.model_dir, config)?;
    let state = AppState::new(translator.clone());
    let app = router(state, args.cors_origin);
    let listener = tokio::net::TcpListener::bind(args.bind)
        .await
        .with_context(|| format!("failed to bind {}", args.bind))?;
    info!(
        bind = %args.bind,
        backend = translator.backend_info().name,
        device = translator.backend_info().device,
        "translation service ready"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("HTTP server failed")?;
    translator.shutdown().await;
    info!("translation service stopped");
    Ok(())
}

fn configure_cpu_threads(args: &Args) -> Result<()> {
    let auto_uses_cpu = cfg!(feature = "cpu")
        && !cfg!(all(
            target_os = "macos",
            target_arch = "aarch64",
            feature = "metal"
        ));
    if !(matches!(args.backend, BackendKind::Cpu)
        || matches!(args.backend, BackendKind::Auto) && auto_uses_cpu)
    {
        return Ok(());
    }
    if !matches!(args.cpu_threads, 1 | 2 | 4) {
        anyhow::bail!("pure Rust cpu_threads must be 1, 2, or 4");
    }

    // SAFETY: this runs at the very start of `main`, before the Tokio runtime,
    // backend owner, matrixmultiply workers, or Rayon global pool are created.
    // Both CPU executors read these process-wide settings on first use.
    unsafe {
        std::env::set_var("MATMUL_NUM_THREADS", args.cpu_threads.to_string());
        std::env::set_var("RAYON_NUM_THREADS", args.cpu_threads.to_string());
    }
    Ok(())
}

fn create_translator(
    backend: BackendKind,
    model_dir: PathBuf,
    config: SchedulerConfig,
) -> Result<Translator> {
    match backend {
        BackendKind::Echo => Translator::start(config, || Ok(EchoBackend)).map_err(Into::into),
        BackendKind::Auto => {
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "metal"))]
            {
                start_metal_translator(config, model_dir)
            }
            #[cfg(all(
                feature = "cpu",
                not(all(target_os = "macos", target_arch = "aarch64", feature = "metal"))
            ))]
            {
                Translator::start(config, move || marian_cpu::CpuModelBackend::load(model_dir))
                    .map_err(Into::into)
            }
            #[cfg(not(any(
                all(target_os = "macos", target_arch = "aarch64", feature = "metal"),
                feature = "cpu"
            )))]
            {
                let _ = model_dir;
                anyhow::bail!(
                    "no production backend is compiled for this platform; enable `metal` on Apple Silicon macOS or `cpu` on any supported platform"
                )
            }
        }
        BackendKind::Metal => {
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "metal"))]
            {
                start_metal_translator(config, model_dir)
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "metal")))]
            {
                let _ = model_dir;
                anyhow::bail!(
                    "Metal backend is not in this binary; build on Apple Silicon with `cargo build --release --features metal` (use --backend echo only for API testing)"
                )
            }
        }
        BackendKind::Cpu => {
            #[cfg(feature = "cpu")]
            {
                Translator::start(config, move || marian_cpu::CpuModelBackend::load(model_dir))
                    .map_err(Into::into)
            }
            #[cfg(not(feature = "cpu"))]
            {
                let _ = model_dir;
                anyhow::bail!(
                    "pure Rust CPU backend is not in this binary; build with `cargo build --release -p marian-server --features cpu`"
                )
            }
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "metal"))]
fn start_metal_translator(config: SchedulerConfig, model_dir: PathBuf) -> Result<Translator> {
    let metal_config = marian_metal::MetalConfig::from_env().map_err(anyhow::Error::msg)?;
    Translator::start(config, move || {
        marian_metal::MetalBackend::load_with_config(model_dir, &metal_config)
    })
    .map_err(Into::into)
}

fn init_tracing(json: bool) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("marian_server=info,tower_http=info"));
    if json {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .json()
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Args, BackendKind};

    #[test]
    fn backend_aliases_are_stable_and_bergamot_is_gone() {
        for alias in ["cpu", "cpu-fp32", "cpu-q8", "rust"] {
            let args = Args::try_parse_from(["marian-edge-server", "--backend", alias]).unwrap();
            assert_eq!(args.backend, BackendKind::Cpu);
        }
        let args = Args::try_parse_from(["marian-edge-server", "--backend", "mlx"]).unwrap();
        assert_eq!(args.backend, BackendKind::Metal);
        assert!(Args::try_parse_from(["marian-edge-server", "--backend", "bergamot"]).is_err());
    }
}
