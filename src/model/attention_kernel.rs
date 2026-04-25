#![allow(dead_code)]

use candle_core::{Device, Result, Tensor};

pub struct PagedAttention {
    pub block_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl PagedAttention {
    pub fn new(block_size: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            block_size,
            num_heads,
            head_dim,
        }
    }

    pub fn forward(
        &self,
        query: &Tensor,
        _key_cache: &Tensor,
        _value_cache: &Tensor,
        _block_table: &Tensor,
        _context_lens: &Tensor,
    ) -> Result<Tensor> {
        let device = query.device();

        match device {
            Device::Cuda(_) => self.software_paged_attention(query),
            _ => self.software_paged_attention(query),
        }
    }

    fn software_paged_attention(&self, query: &Tensor) -> Result<Tensor> {
        Ok(query.clone())
    }
}
