use std::sync::Arc;

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::distributed::DistributedContext;
use crate::model::config::LlamaConfig;
use crate::model::kv_cache::CacheContext;
use crate::model::layers::{RmsNorm, RotaryEmbedding};
use crate::model::pipeline::PipelineContext;

pub struct Qwen2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rope: RotaryEmbedding,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Qwen2Attention {
    pub fn new(
        cfg: &LlamaConfig,
        vb: VarBuilder,
        device: &Device,
        _dist: Arc<DistributedContext>,
    ) -> Result<Self> {
        let world_size = _dist.world_size as usize;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let n_heads = cfg.num_attention_heads / world_size;
        let n_kv_heads = cfg.num_key_value_heads / world_size;

        let q_proj = candle_nn::linear(cfg.hidden_size, n_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(cfg.hidden_size, n_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(cfg.hidden_size, n_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear(n_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let rope = RotaryEmbedding::new(head_dim, cfg.max_seq_len.unwrap_or(32768), device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            n_heads,
            n_kv_heads,
            head_dim,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        index: usize,
        mut cache: Option<&mut CacheContext>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?;

        let q = self.rope.apply(&q, index)?;
        let k = self.rope.apply(&k, index)?;

        let (full_k, full_v) = if let Some(ctx) = cache.as_mut() {
            let rid = ctx.request_id;
            ctx.manager.append_kv(rid, &k, &v)?;
            let ctx_len = ctx.manager.get_context_len(rid);
            if seq_len == 1 && ctx_len > seq_len {
                let cached_k = ctx
                    .manager
                    .get_cached_key(rid)
                    .ok_or_else(|| candle_core::Error::Msg("cached key missing".into()))?;
                let cached_v = ctx
                    .manager
                    .get_cached_value(rid)
                    .ok_or_else(|| candle_core::Error::Msg("cached value missing".into()))?;
                (cached_k, cached_v)
            } else {
                (k.clone(), v.clone())
            }
        } else {
            (k.clone(), v.clone())
        };

        let full_k = expand_kv(&full_k, self.n_heads)?;
        let full_v = expand_kv(&full_v, self.n_heads)?;

        let att = (q.matmul(&full_k.transpose(2, 3)?)? / (self.head_dim as f64).sqrt())?;
        let att = att.narrow(3, full_k.dim(2)? - seq_len, seq_len)?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let out = att.matmul(&full_v.narrow(1, full_v.dim(1)? - seq_len, seq_len)?)?;
        let out = out.reshape((b_sz, seq_len, self.n_heads * self.head_dim))?;

        self.o_proj.forward(&out)
    }
}

fn expand_kv(tensor: &Tensor, n_heads: usize) -> Result<Tensor> {
    let kv_heads = tensor.dim(2)?;
    if kv_heads == n_heads {
        return Ok(tensor.clone());
    }
    let group_size = n_heads / kv_heads;
    let mut heads = Vec::with_capacity(n_heads);
    for h in 0..kv_heads {
        let h_t = tensor.narrow(2, h, 1)?;
        for _ in 0..group_size {
            heads.push(h_t.clone());
        }
    }
    Tensor::cat(&heads, 2)
}

pub struct Qwen2MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen2MLP {
    pub fn new(cfg: &LlamaConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj =
            candle_nn::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            candle_nn::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.gate_proj.forward(x)?)? * self.up_proj.forward(x)?)?;
        self.down_proj.forward(&x)
    }
}

pub struct Qwen2DecoderLayer {
    self_attn: Qwen2Attention,
    mlp: Qwen2MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen2DecoderLayer {
    pub fn new(
        cfg: &LlamaConfig,
        vb: VarBuilder,
        device: &Device,
        dist: Arc<DistributedContext>,
    ) -> Result<Self> {
        let self_attn = Qwen2Attention::new(cfg, vb.pp("self_attn"), device, dist)?;
        let mlp = Qwen2MLP::new(cfg, vb.pp("mlp"))?;
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

    pub fn forward(
        &self,
        x: &Tensor,
        index: usize,
        cache: Option<&mut CacheContext>,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self.self_attn.forward(&x, index, cache)? + residual)?;
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = (self.mlp.forward(&x)? + residual)?;
        Ok(x)
    }
}

pub struct Qwen2Model {
    embed_tokens: Option<candle_nn::Embedding>,
    layers: Vec<Qwen2DecoderLayer>,
    norm: Option<RmsNorm>,
    lm_head: Option<Linear>,
    #[allow(dead_code)]
    pipeline_ctx: PipelineContext,
}

impl Qwen2Model {
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
            layers.push(Qwen2DecoderLayer::new(
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
            pipeline_ctx,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        index: usize,
        mut cache: Option<&mut CacheContext>,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        if let Some(embed) = &self.embed_tokens {
            x = embed.forward(&x)?;
        }
        for layer in &self.layers {
            x = layer.forward(&x, index, cache.as_deref_mut())?;
        }
        if let Some(norm) = &self.norm {
            x = norm.forward(&x)?;
        }
        if let Some(lm_head) = &self.lm_head {
            x = lm_head.forward(&x)?;
        }
        Ok(x)
    }
}
