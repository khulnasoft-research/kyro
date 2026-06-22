mod api;
mod config;
mod device;
mod distributed;
mod metrics;
mod model;
mod scheduler;
mod speculative;
mod worker;

use crate::api::tokenizer::LuminaTokenizer;
use crate::config::Cli;
use crate::distributed::DistributedContext;
use crate::model::loader::{LoadedModel, ModelLoader};
use crate::scheduler::block_manager::BlockManager;
use crate::scheduler::continuous_batching::Scheduler;
use crate::worker::Worker;
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Notify, RwLock};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging early so startup messages are captured
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive(Level::INFO.into()),
        )
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Parse CLI arguments
    let cli = Cli::parse_or_default();
    info!(?cli, "Starting Kyro LLM Engine");

    // 1. Hardware detection
    let device = device::get_device()?;
    info!(device = ?device, "Hardware detected");

    // 2. Initialize Distributed Context, Block Manager, Scheduler, and Metrics
    let dist = Arc::new(DistributedContext::new());
    info!(
        rank = dist.rank,
        world_size = dist.world_size,
        "Distributed context initialized"
    );

    let block_manager = BlockManager::new(cli.block_size, cli.num_gpu_blocks, cli.num_cpu_blocks);
    let scheduler_cfg = scheduler::continuous_batching::SchedulerConfig {
        max_tokens_per_iter: cli.max_tokens_per_iter,
        max_prefill_chunk_size: cli.max_prefill_chunk_size,
        request_timeout_secs: cli.request_timeout_secs,
    };
    let scheduler = Arc::new(RwLock::new(Scheduler::new(block_manager, scheduler_cfg)));
    let notify = Arc::new(Notify::new());

    let prometheus_registry = prometheus::Registry::new();
    let metrics = metrics::EngineMetrics::new(&prometheus_registry)?;
    info!("Scheduler and metrics initialized");

    // 3. Load Model — from CLI arg or env var, or fallback to dummy
    let loaded_model = if let Some(ref mp) = cli.model_path {
        info!(path = %mp.display(), "Loading model");
        let loader = ModelLoader::new(mp).context("Failed to initialize model loader")?;
        loader
            .load(&device, dist.clone())
            .context("Failed to load model")?
    } else {
        info!("No model path provided; using mock model");
        let cfg = model::config::LlamaConfig::llama_7b();
        LoadedModel::Standard(model::llama::LlamaModel::dummy(&cfg)?)
    };
    info!("Model loaded");

    // 3b. Load Tokenizer
    let tokenizer_path: Option<PathBuf> = cli.tokenizer_path.or_else(|| {
        cli.model_path.as_ref().map(|mp| {
            let candidate = mp.join("tokenizer.json");
            if candidate.exists() {
                candidate
            } else {
                mp.to_path_buf()
            }
        })
    });

    let tokenizer = tokenizer_path
        .as_ref()
        .filter(|p| p.join("tokenizer.json").exists() || p.extension().is_some_and(|e| e == "json"))
        .and_then(|tp| {
            let path = if tp.is_dir() {
                tp.join("tokenizer.json")
            } else {
                tp.clone()
            };
            match LuminaTokenizer::from_file(&path) {
                Ok(tk) => {
                    info!(path = %path.display(), "Tokenizer loaded");
                    Some(tk)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to load tokenizer, using fallback");
                    None
                }
            }
        });

    if tokenizer.is_none() {
        info!("No tokenizer loaded; using fallback token format");
    }

    // 4. Start Worker Loop
    let mut worker = Worker::new(loaded_model, scheduler.clone(), device, metrics);
    let worker_notify = notify.clone();
    tokio::spawn(async move {
        if let Err(e) = worker.run_loop(worker_notify).await {
            tracing::error!(error = %e, "Worker loop failed");
        }
    });

    // 5. Build API Router
    let app_state = Arc::new(api::openai::AppState::new(
        scheduler,
        notify,
        prometheus_registry,
        tokenizer,
    ));
    let app = api::openai::app(app_state);

    // 6. Start API Server with graceful shutdown
    let addr = format!("{}:{}", cli.host, cli.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!(addr = %addr, "Kyro API serving");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => { info!("Received Ctrl+C, shutting down"); }
        () = terminate => { info!("Received SIGTERM, shutting down"); }
    }
}
