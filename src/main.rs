mod api;
mod device;
mod distributed;
mod metrics;
mod model;
mod scheduler;
mod speculative;
mod worker;

use crate::distributed::DistributedContext;
use crate::model::loader::{LoadedModel, ModelLoader};
use crate::scheduler::block_manager::BlockManager;
use crate::scheduler::continuous_batching::Scheduler;
use crate::worker::Worker;
use anyhow::{Context, Result};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{RwLock, Notify};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive(Level::INFO.into()),
        )
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting Kyro LLM Engine...");

    // 1. Hardware detection
    let device = device::get_device()?;
    info!("Using device: {:?}", device);

    // 2. Initialize Distributed Context, Block Manager, Scheduler, and Metrics
    let dist = Arc::new(DistributedContext::new());
    info!(
        "Distributed context initialized (Rank: {}, World Size: {})",
        dist.rank, dist.world_size
    );

    let block_manager = BlockManager::new(16, 1024, 256);
    let scheduler_cfg = scheduler::continuous_batching::SchedulerConfig::default();
    let scheduler = Arc::new(RwLock::new(Scheduler::new(block_manager, scheduler_cfg)));
    let notify = Arc::new(Notify::new());

    let registry = prometheus::Registry::new();
    let metrics = metrics::EngineMetrics::new(&registry)?;
    info!("Scheduler and Metrics initialized.");

    // 3. Load Model — from CLI arg `--model-path <PATH>`, env `KYRO_MODEL_PATH`, or fallback to dummy
    let model_path: Option<PathBuf> = {
        let mut args = env::args().peekable();
        let mut path = None;
        while let Some(arg) = args.next() {
            if arg == "--model-path" {
                path = args.next().map(PathBuf::from);
            }
        }
        path.or_else(|| env::var("KYRO_MODEL_PATH").ok().map(PathBuf::from))
    };

    let loaded_model = if let Some(ref mp) = model_path {
        info!("Loading model from: {}", mp.display());
        let loader = ModelLoader::new(mp).context("Failed to initialize model loader")?;
        loader.load(&device, dist.clone()).context("Failed to load model")?
    } else {
        info!("No model path provided; using mock model.");
        let cfg = model::config::LlamaConfig::llama_7b();
        LoadedModel::Standard(model::llama::LlamaModel::dummy(&cfg)?)
    };
    info!("Model loaded.");

    // 4. Start Worker Loop
    let mut worker = Worker::new(loaded_model, scheduler.clone(), device, metrics);
    let worker_notify = notify.clone();
    tokio::spawn(async move {
        if let Err(e) = worker.run_loop(worker_notify).await {
            tracing::error!("Worker loop failed: {:?}", e);
        }
    });

    // 5. Start API Server
    let app_state = Arc::new(api::openai::AppState::new(scheduler, notify));
    let app = api::openai::app(app_state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    info!("Kyro API serving on http://localhost:3000");

    axum::serve(listener, app).await?;

    Ok(())
}
