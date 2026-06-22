use candle_core::{DType, Result, Tensor, D};

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
        self.paged_attention_kernel(query, key_cache, value_cache, block_table, context_lens)
    }

    /// Expand KV heads from num_kv_heads to num_heads for GQA by repeating each KV head.
    fn expand_kv_for_gqa(&self, tensor: &Tensor) -> Result<Tensor> {
        let kv_heads = tensor.dim(0)?;
        if kv_heads == self.num_heads {
            return Ok(tensor.clone());
        }
        let group_size = self.num_heads / kv_heads;
        let mut heads = Vec::with_capacity(self.num_heads);
        for h in 0..kv_heads {
            let h_t = tensor.get(h)?;
            for _ in 0..group_size {
                heads.push(h_t.unsqueeze(0)?);
            }
        }
        Tensor::cat(&heads, 0)
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
                outputs.push(Tensor::zeros(
                    (1, self.num_heads, self.head_dim),
                    DType::F32,
                    device,
                )?);
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
                let k_block = self.expand_kv_for_gqa(&k_block)?;

                let k_t = k_block.transpose(1, 2)?;
                let q_2d = q.squeeze(0)?;
                let q_exp = q_2d.unsqueeze(1)?;
                let mut scores = q_exp.matmul(&k_t)?.squeeze(1)?;
                scores = (scores * scale)?;

                let scores_2d: Vec<Vec<f32>> = scores.to_vec2::<f32>()?;
                for h_scores in scores_2d.iter().take(self.num_heads) {
                    attn_scores.extend_from_slice(&h_scores[..block_len]);
                }
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
                let v_block = self.expand_kv_for_gqa(&v_block)?;
                let w = attn_weights.narrow(D::Minus1, start_token, block_len)?;

                let v_narrow = v_block.narrow(1, 0, block_len)?;
                let w_exp = w.unsqueeze(1)?;
                let contrib = w_exp.matmul(&v_narrow)?.squeeze(1)?;
                output = (output + contrib.unsqueeze(0)?)?;
            }

            outputs.push(output);
        }

        Tensor::cat(&outputs, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn make_attention(
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        num_kv_heads: usize,
    ) -> PagedAttention {
        PagedAttention::new(block_size, num_heads, head_dim, num_kv_heads)
    }

    #[test]
    fn test_paged_attention_creation() {
        let attn = make_attention(16, 8, 64, 2);
        assert_eq!(attn.block_size, 16);
        assert_eq!(attn.num_heads, 8);
        assert_eq!(attn.head_dim, 64);
        assert_eq!(attn.num_kv_heads, 2);
    }

    #[test]
    fn test_paged_attention_zero_context() {
        let device = Device::Cpu;
        let attn = make_attention(16, 8, 64, 2);
        let query = Tensor::zeros((1, 8, 64), DType::F32, &device).unwrap();
        let key_cache = Tensor::zeros((4, 2, 16, 64), DType::F32, &device).unwrap();
        let value_cache = Tensor::zeros((4, 2, 16, 64), DType::F32, &device).unwrap();
        let block_table = Tensor::zeros((1, 4), DType::I64, &device).unwrap();
        let context_lens = Tensor::zeros((1,), DType::I64, &device).unwrap();

        let out = attn
            .forward(
                &query,
                &key_cache,
                &value_cache,
                &block_table,
                &context_lens,
            )
            .unwrap();
        assert_eq!(out.dims(), &[1, 8, 64]);
    }

    #[test]
    fn test_paged_attention_mha_single_block() {
        let device = Device::Cpu;
        let attn = make_attention(16, 8, 64, 8);
        let query = Tensor::ones((1, 8, 64), DType::F32, &device).unwrap();
        let key_cache = Tensor::ones((4, 8, 16, 64), DType::F32, &device).unwrap();
        let value_cache = Tensor::ones((4, 8, 16, 64), DType::F32, &device).unwrap();
        let block_table = Tensor::zeros((1, 4), DType::I64, &device).unwrap();
        let context_lens = Tensor::full(8i64, (1,), &device).unwrap();

        let out = attn
            .forward(
                &query,
                &key_cache,
                &value_cache,
                &block_table,
                &context_lens,
            )
            .unwrap();
        assert_eq!(out.dims(), &[1, 8, 64]);
    }

    #[test]
    fn test_paged_attention_mha_multi_block() {
        let device = Device::Cpu;
        let attn = make_attention(16, 8, 64, 8);
        let query = Tensor::ones((1, 8, 64), DType::F32, &device).unwrap();
        let key_cache = Tensor::ones((4, 8, 16, 64), DType::F32, &device).unwrap();
        let value_cache = Tensor::ones((4, 8, 16, 64), DType::F32, &device).unwrap();
        let block_table = Tensor::full(0i64, (1, 4), &device).unwrap();
        let context_lens = Tensor::full(24i64, (1,), &device).unwrap();

        let out = attn
            .forward(
                &query,
                &key_cache,
                &value_cache,
                &block_table,
                &context_lens,
            )
            .unwrap();
        assert_eq!(out.dims(), &[1, 8, 64]);
    }
}
