use anyhow::Result;
use prometheus::{Counter, Gauge, Histogram, Registry};
use std::sync::Arc;

pub struct EngineMetrics {
    #[allow(dead_code)]
    pub total_requests: Counter,
    pub total_tokens_generated: Counter,
    #[allow(dead_code)]
    pub token_latency: Histogram,
    #[allow(dead_code)]
    pub time_to_first_token: Histogram,
    #[allow(dead_code)]
    pub time_between_tokens: Histogram,
    #[allow(dead_code)]
    pub kv_cache_usage: Gauge,
}

impl EngineMetrics {
    pub fn new(registry: &Registry) -> Result<Arc<Self>> {
        let total_requests = prometheus::register_counter_with_registry!(
            "kyro_requests_total",
            "Total number of requests processed",
            registry
        )?;

        let total_tokens_generated = prometheus::register_counter_with_registry!(
            "kyro_tokens_total",
            "Total number of tokens generated",
            registry
        )?;

        let token_latency = prometheus::register_histogram_with_registry!(
            "kyro_token_latency_seconds",
            "Latency per token generation",
            registry
        )?;

        let time_to_first_token = prometheus::register_histogram_with_registry!(
            "kyro_ttft_ms",
            "Time to first token",
            registry
        )?;

        let time_between_tokens = prometheus::register_histogram_with_registry!(
            "kyro_tbt_ms",
            "Time between tokens",
            registry
        )?;

        let kv_cache_usage = prometheus::register_gauge_with_registry!(
            "kyro_kv_cache_usage_percent",
            "KV cache utilization percentage",
            registry
        )?;

        Ok(Arc::new(Self {
            total_requests,
            total_tokens_generated,
            token_latency,
            time_to_first_token,
            time_between_tokens,
            kv_cache_usage,
        }))
    }
}
