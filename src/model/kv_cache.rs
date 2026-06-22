use candle_core::{Device, Result, Tensor};

#[allow(dead_code)]
pub struct KVCache {
    pub key_cache: Tensor,
    pub value_cache: Tensor,
    pub block_size: usize,
    pub num_blocks: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

#[allow(dead_code)]
impl KVCache {
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let key_cache = Tensor::zeros(
            (num_blocks, num_kv_heads, block_size, head_dim),
            candle_core::DType::F16,
            device,
        )?;
        let value_cache = Tensor::zeros(
            (num_blocks, num_kv_heads, block_size, head_dim),
            candle_core::DType::F16,
            device,
        )?;
        Ok(Self {
            key_cache,
            value_cache,
            block_size,
            num_blocks,
            num_kv_heads,
            head_dim,
        })
    }
}
