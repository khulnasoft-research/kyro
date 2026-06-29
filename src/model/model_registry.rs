use std::sync::Arc;

use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::distributed::DistributedContext;
use crate::model::config::{LlamaConfig, ModelArchitecture};
use crate::model::kv_cache::CacheContext;
use crate::model::llama::LlamaModel;
use crate::model::loader::ModelForward;
use crate::model::pipeline::PipelineContext;
use crate::model::qwen::Qwen2Model;

pub enum ModelInstance {
    Llama(LlamaModel),
    Qwen2(Qwen2Model),
}

impl ModelForward for ModelInstance {
    fn forward(
        &mut self,
        x: &Tensor,
        index: usize,
        cache: Option<&mut CacheContext>,
    ) -> Result<Tensor> {
        match self {
            ModelInstance::Llama(m) => m.forward(x, index, cache),
            ModelInstance::Qwen2(m) => m.forward(x, index, cache),
        }
    }
}

impl ModelInstance {
    pub fn from_architecture(
        architecture: &ModelArchitecture,
        config: &LlamaConfig,
        vb: VarBuilder,
        device: &Device,
        dist: Arc<DistributedContext>,
        pipeline_ctx: PipelineContext,
    ) -> Result<Self> {
        match architecture {
            ModelArchitecture::DecoderOnly => {
                let model = LlamaModel::new(config, vb, device, dist, pipeline_ctx)?;
                Ok(ModelInstance::Llama(model))
            }
            ModelArchitecture::MixtureOfExperts => {
                let model = Qwen2Model::new(config, vb, device, dist, pipeline_ctx)?;
                Ok(ModelInstance::Qwen2(model))
            }
            _ => {
                let model = LlamaModel::new(config, vb, device, dist, pipeline_ctx)?;
                Ok(ModelInstance::Llama(model))
            }
        }
    }
}
