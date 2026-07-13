use std::{net::SocketAddr, path::PathBuf, time::Duration};

use anyhow::{Context, Result};
use axum::http::HeaderValue;
use clap::{Parser, ValueEnum};
use marian_core::{EchoBackend, SchedulerConfig, Translator};
use marian_server::{AppState, router};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendKind {
    Auto,
    Mlx,
    Bergamot,
    Echo,
}

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    #[arg(long, env = "MARIAN_MLX_BIND", default_value = "127.0.0.1:3000")]
    bind: SocketAddr,

    #[arg(long, env = "MARIAN_MLX_BACKEND", value_enum, default_value = "auto")]
    backend: BackendKind,

    #[arg(long, env = "MARIAN_MLX_MODEL_DIR", default_value = "models/enzh")]
    model_dir: PathBuf,

    #[arg(
        long,
        env = "MARIAN_MLX_BERGAMOT_WORKER",
        default_value = "/usr/local/bin/marian-mlx-bergamot-worker"
    )]
    bergamot_worker: PathBuf,

    #[arg(long, env = "MARIAN_MLX_CPU_THREADS", default_value_t = 1)]
    cpu_threads: usize,

    #[arg(long, env = "MARIAN_MLX_QUEUE_CAPACITY", default_value_t = 256)]
    queue_capacity: usize,

    #[arg(long, env = "MARIAN_MLX_MAX_BATCH_SIZE", default_value_t = 16)]
    max_batch_size: usize,

    #[arg(
        long,
        env = "MARIAN_MLX_MAX_PADDED_SOURCE_CHARS",
        default_value_t = 4096
    )]
    max_padded_source_chars: usize,

    #[arg(long, env = "MARIAN_MLX_BATCH_WINDOW_US", default_value_t = 750)]
    batch_window_us: u64,

    #[arg(long, env = "MARIAN_MLX_REQUEST_TIMEOUT_MS", default_value_t = 30_000)]
    request_timeout_ms: u64,

    #[arg(long, env = "MARIAN_MLX_CORS_ORIGIN")]
    cors_origin: Option<HeaderValue>,

    #[arg(long, env = "MARIAN_MLX_JSON_LOGS", default_value_t = false)]
    json_logs: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    init_tracing(args.json_logs);
    let config = SchedulerConfig {
        queue_capacity: args.queue_capacity,
        max_batch_size: args.max_batch_size,
        max_padded_source_chars: args.max_padded_source_chars,
        batch_window: Duration::from_micros(args.batch_window_us),
        request_timeout: Duration::from_millis(args.request_timeout_ms),
        ..SchedulerConfig::default()
    };

    let translator = create_translator(
        args.backend,
        args.model_dir,
        args.bergamot_worker,
        args.cpu_threads,
        config,
    )?;
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

fn create_translator(
    backend: BackendKind,
    model_dir: PathBuf,
    bergamot_worker: PathBuf,
    cpu_threads: usize,
    config: SchedulerConfig,
) -> Result<Translator> {
    match backend {
        BackendKind::Echo => Translator::start(config, || Ok(EchoBackend)).map_err(Into::into),
        BackendKind::Auto => {
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
            {
                let _ = (bergamot_worker, cpu_threads);
                return Translator::start(config, move || marian_mlx::MlxBackend::load(model_dir))
                    .map_err(Into::into);
            }
            #[cfg(all(target_os = "linux", feature = "bergamot"))]
            {
                return Translator::start(config, move || {
                    marian_bergamot::BergamotBackend::load(model_dir, bergamot_worker, cpu_threads)
                })
                .map_err(Into::into);
            }
            #[cfg(not(any(
                all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
                all(target_os = "linux", feature = "bergamot")
            )))]
            {
                let _ = (model_dir, bergamot_worker, cpu_threads);
                anyhow::bail!(
                    "no production backend is compiled for this platform; enable `mlx` on Apple Silicon macOS or `bergamot` on Linux"
                )
            }
        }
        BackendKind::Mlx => {
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
            {
                Translator::start(config, move || marian_mlx::MlxBackend::load(model_dir))
                    .map_err(Into::into)
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "mlx")))]
            {
                let _ = model_dir;
                anyhow::bail!(
                    "MLX backend is not in this binary; build on Apple Silicon with `cargo build --release --features mlx` (use --backend echo only for API testing)"
                )
            }
        }
        BackendKind::Bergamot => {
            #[cfg(all(target_os = "linux", feature = "bergamot"))]
            {
                Translator::start(config, move || {
                    marian_bergamot::BergamotBackend::load(model_dir, bergamot_worker, cpu_threads)
                })
                .map_err(Into::into)
            }
            #[cfg(not(all(target_os = "linux", feature = "bergamot")))]
            {
                let _ = (model_dir, bergamot_worker, cpu_threads);
                anyhow::bail!(
                    "Bergamot backend is not in this binary; build on Linux with `cargo build --release --features bergamot`"
                )
            }
        }
    }
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
