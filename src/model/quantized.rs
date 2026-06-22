use crate::model::kv_cache::CacheContext;
use candle_core::{Device, Result, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use std::path::Path;

#[allow(dead_code)]
pub struct QuantizedLlama {
    pub inner: ModelWeights,
    pub device: Device,
}

impl QuantizedLlama {
    #[allow(dead_code)]
    pub fn load_gguf<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(path.as_ref())?;
        let model = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        let inner = ModelWeights::from_gguf(model, &mut file, device)?;
        Ok(Self {
            inner,
            device: device.clone(),
        })
    }

    #[allow(dead_code)]
    pub fn forward(
        &mut self,
        x: &Tensor,
        index: usize,
        _cache: Option<&mut CacheContext>,
    ) -> Result<Tensor> {
        self.inner.forward(x, index)
    }
}
