use crate::distributed::DistributedContext;
use crate::model::attention_kernel::PagedAttention;
use crate::model::config::LlamaConfig;
use crate::model::layers::{RmsNorm, RotaryEmbedding};
use crate::model::pipeline::PipelineContext;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use std::sync::Arc;

pub struct LlamaAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    rope: RotaryEmbedding,
    n_heads: usize,
    n_kv_heads: usize,
    pub head_dim: usize,
    pub dist: Arc<DistributedContext>,
    pub paged_attn: PagedAttention,
}

impl LlamaAttention {
    pub fn new(
        cfg: &LlamaConfig,
        vb: VarBuilder,
        device: &Device,
        dist: Arc<DistributedContext>,
    ) -> Result<Self> {
        let world_size = dist.world_size as usize;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let n_heads = cfg.num_attention_heads / world_size;
        let n_kv_heads = cfg.num_key_value_heads / world_size;

        let q_proj = candle_nn::linear(cfg.hidden_size, n_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(cfg.hidden_size, n_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(cfg.hidden_size, n_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear(n_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let rope = RotaryEmbedding::new(head_dim, 4096, device)?;
        let paged_attn = PagedAttention::new(16, n_heads, head_dim); // 16 is block size

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            n_heads,
            n_kv_heads,
            head_dim,
            dist,
            paged_attn,
        })
    }

    pub fn forward(&self, x: &Tensor, index: usize) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?;

        let q = self.rope.apply(&q, index)?;
        let k = self.rope.apply(&k, index)?;

        // SDPA (Simplified)
        let att = (q.matmul(&k.transpose(2, 3)?)? / (self.head_dim as f64).sqrt())?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let out = att.matmul(&v)?;
        let out = out.reshape((b_sz, seq_len, self.n_heads * self.head_dim))?;

        // Row-parallel: partial sum
        let out = self.o_proj.forward(&out)?;

        // Tensor Parallelism: All-Reduce to synchronize GPUs
        // self.dist.all_reduce(&out)
        Ok(out)
    }
}

pub struct LlamaMLP {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
    dist: Arc<DistributedContext>,
}

impl LlamaMLP {
    pub fn new(cfg: &LlamaConfig, vb: VarBuilder, dist: Arc<DistributedContext>) -> Result<Self> {
        let world_size = dist.world_size as usize;
        let intermediate_size = cfg.intermediate_size / world_size;

        let gate_proj = candle_nn::linear(cfg.hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear(cfg.hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear(intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            dist,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.gate_proj.forward(x)?)? * self.up_proj.forward(x)?)?;
        let out = self.down_proj.forward(&x)?;

        // Tensor Parallelism: All-Reduce to synchronize GPUs
        // self.dist.all_reduce(&out)
        Ok(out)
    }
}

pub struct LlamaDecoderLayer {
    self_attn: LlamaAttention,
    mlp: LlamaMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl LlamaDecoderLayer {
    pub fn new(
        cfg: &LlamaConfig,
        vb: VarBuilder,
        device: &Device,
        dist: Arc<DistributedContext>,
    ) -> Result<Self> {
        let self_attn = LlamaAttention::new(cfg, vb.pp("self_attn"), device, dist.clone())?;
        let mlp = LlamaMLP::new(cfg, vb.pp("mlp"), dist.clone())?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(&self, x: &Tensor, index: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self.self_attn.forward(&x, index)? + residual)?;
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = (self.mlp.forward(&x)? + residual)?;
        Ok(x)
    }
}

pub struct LlamaModel {
    embed_tokens: Option<candle_nn::Embedding>,
    layers: Vec<LlamaDecoderLayer>,
    norm: Option<RmsNorm>,
    lm_head: Option<candle_nn::Linear>,
    device: Device,
    pipeline_ctx: PipelineContext,
}

impl LlamaModel {
    pub fn new(
        cfg: &LlamaConfig,
        vb: VarBuilder,
        device: &Device,
        dist: Arc<DistributedContext>,
        pipeline_ctx: PipelineContext,
    ) -> Result<Self> {
        let embed_tokens = if pipeline_ctx.is_first_stage() {
            Some(candle_nn::embedding(
                cfg.vocab_size,
                cfg.hidden_size,
                vb.pp("model.embed_tokens"),
            )?)
        } else {
            None
        };

        let mut layers = Vec::new();
        let vb_l = vb.pp("model.layers");
        for layer_idx in pipeline_ctx.start_layer..pipeline_ctx.end_layer {
            layers.push(LlamaDecoderLayer::new(
                cfg,
                vb_l.pp(layer_idx),
                device,
                dist.clone(),
            )?);
        }

        let norm = if pipeline_ctx.is_last_stage() {
            Some(RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("model.norm"),
            )?)
        } else {
            None
        };

        let lm_head = if pipeline_ctx.is_last_stage() {
            Some(candle_nn::linear(
                cfg.hidden_size,
                cfg.vocab_size,
                vb.pp("lm_head"),
            )?)
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            pipeline_ctx,
        })
    }

    pub fn forward(&self, x: &Tensor, index: usize) -> Result<Tensor> {
        let mut x = x.clone();

        if let Some(embed) = &self.embed_tokens {
            x = embed.forward(&x)?;
        }

        for layer in &self.layers {
            x = layer.forward(&x, index)?;
        }

        if let Some(norm) = &self.norm {
            x = norm.forward(&x)?;
        }
        if let Some(lm_head) = &self.lm_head {
            x = lm_head.forward(&x)?;
        }

        Ok(x)
    }

    pub fn dummy(cfg: &LlamaConfig) -> Result<Self> {
        Ok(Self {
            embed_tokens: None,
            layers: Vec::new(),
            norm: None,
            lm_head: None,
            device: Device::Cpu,
            pipeline_ctx: PipelineContext::new(0, 1, cfg.num_hidden_layers),
        })
    }
}
