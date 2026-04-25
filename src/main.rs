mod device;
mod model;
mod scheduler;
mod api;
mod worker;
mod speculative;
mod metrics;
mod distributed;

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use crate::scheduler::block_manager::BlockManager;
use crate::scheduler::continuous_batching::Scheduler;
use crate::worker::Worker;
use crate::distributed::DistributedContext;
use crate::model::pipeline::PipelineContext;
use crate::model::loader::LoadedModel;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env()
            .add_directive(Level::INFO.into()))
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting Kyro LLM Engine...");

    // 1. Hardware detection
    let device = device::get_device()?;
    info!("Using device: {:?}", device);

    // 2. Initialize Distributed Context, Block Manager, Scheduler, and Metrics
    let dist = Arc::new(DistributedContext::new());
    info!("Distributed context initialized (Rank: {}, World Size: {})", dist.rank, dist.world_size);

    let block_manager = BlockManager::new(16, 1024, 256);
    let scheduler_cfg = scheduler::continuous_batching::SchedulerConfig::default();
    let scheduler = Arc::new(Mutex::new(Scheduler::new(block_manager, scheduler_cfg)));
    
    let registry = prometheus::Registry::new();
    let metrics = metrics::EngineMetrics::new(&registry)?;
    info!("Scheduler and Metrics initialized.");

    // 3. Load Model (Mock for now, or use actual loader if path provided)
    let cfg = model::config::LlamaConfig::llama_7b();
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[], candle_core::DType::F16, &device).unwrap_or_else(|_| {
            // Fallback to zeros for testing if no files
            candle_nn::VarBuilder::from_slice(&[], candle_core::DType::F16, &device)
        })
    };
    
    let pipeline_ctx = PipelineContext::new(dist.rank, dist.world_size, cfg.num_hidden_layers);
    let model = model::llama::LlamaModel::new(&cfg, vb, &device, dist.clone(), pipeline_ctx)?;
    let loaded_model = LoadedModel::Standard(model);
    info!("Model loaded.");

    // 4. Start Worker Loop
    let mut worker = Worker::new(loaded_model, scheduler.clone(), device, metrics);
    tokio::spawn(async move {
        if let Err(e) = worker.run_loop().await {
            tracing::error!("Worker loop failed: {:?}", e);
        }
    });

    // 5. Start API Server
    let app = api::openai::app(scheduler);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    info!("Kyro API serving on http://localhost:3000");
    
    axum::serve(listener, app).await?;

    Ok(())
}
