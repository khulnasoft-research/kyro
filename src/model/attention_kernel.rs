use candle_core::{D, DType, Device, Result, Tensor};

#[allow(dead_code)]
pub struct PagedAttention {
    pub block_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_kv_heads: usize,
}

impl PagedAttention {
    pub fn new(block_size: usize, num_heads: usize, head_dim: usize, num_kv_heads: usize) -> Self {
        Self {
            block_size,
            num_heads,
            head_dim,
            num_kv_heads,
        }
    }

    #[allow(dead_code)]
    pub fn forward(
        &self,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        block_table: &Tensor,
        context_lens: &Tensor,
    ) -> Result<Tensor> {
        let device = query.device();
        match device {
            Device::Cuda(_) => self.paged_attention_kernel(query, key_cache, value_cache, block_table, context_lens),
            _ => self.paged_attention_kernel(query, key_cache, value_cache, block_table, context_lens),
        }
    }

    /// Software PagedAttention: block-sparse attention with cached K/V blocks.
    ///
    /// Shapes:
    ///   query:        [batch_size, num_heads, head_dim]
    ///   key_cache:    [num_blocks, num_kv_heads, block_size, head_dim]
    ///   value_cache:  [num_blocks, num_kv_heads, block_size, head_dim]
    ///   block_table:  [batch_size, max_num_blocks_per_seq]  (i64 indices)
    ///   context_lens: [batch_size]  (i64, number of cached tokens per sequence)
    #[allow(dead_code)]
    fn paged_attention_kernel(
        &self,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        block_table: &Tensor,
        context_lens: &Tensor,
    ) -> Result<Tensor> {
        let batch_size = query.dims()[0];
        let device = query.device();
        let scale = 1.0 / (self.head_dim as f64).sqrt();

        let mut outputs = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let q = query.get(b)?.unsqueeze(0)?;
            let ctx_len = context_lens.get(b)?.to_scalar::<i64>()? as usize;

            if ctx_len == 0 {
                outputs.push(Tensor::zeros((1, self.num_heads, self.head_dim), DType::F32, device)?);
                continue;
            }

            let num_blocks = ctx_len.div_ceil(self.block_size);
            let mut attn_scores: Vec<f32> = Vec::with_capacity(ctx_len);

            for block_idx in 0..num_blocks {
                let tbl = block_table.get(b)?.get(block_idx)?.to_scalar::<i64>()? as usize;
                let start_token = block_idx * self.block_size;
                let end_token = std::cmp::min(start_token + self.block_size, ctx_len);
                let block_len = end_token - start_token;

                let k_block = key_cache.get(tbl)?.squeeze(0)?;
                let q_expanded = q.transpose(1, 2)?;

                let scores = q_expanded.matmul(&k_block)?;
                let scores = (scores * scale)?;

                let scores_flat = scores.flatten_all()?.to_vec1::<f32>()?;
                let block_scores = &scores_flat[scores_flat.len() - block_len * self.num_heads..];
                attn_scores.extend_from_slice(block_scores);
            }

            let attn_t = Tensor::from_slice(&attn_scores, (ctx_len, self.num_heads), device)?;
            let attn_t = attn_t.transpose(0, 1)?;
            let attn_weights = candle_nn::ops::softmax(&attn_t, D::Minus1)?;

            let mut output = Tensor::zeros((1, self.num_heads, self.head_dim), DType::F32, device)?;

            for block_idx in 0..num_blocks {
                let tbl = block_table.get(b)?.get(block_idx)?.to_scalar::<i64>()? as usize;
                let start_token = block_idx * self.block_size;
                let end_token = std::cmp::min(start_token + self.block_size, ctx_len);
                let block_len = end_token - start_token;

                let v_block = value_cache.get(tbl)?.squeeze(0)?;
                let w = attn_weights.narrow(D::Minus1, start_token, block_len)?;
                output = (output + w.matmul(&v_block)?)?;
            }

            outputs.push(output);
        }

        Tensor::cat(&outputs, 0)
    }
}
