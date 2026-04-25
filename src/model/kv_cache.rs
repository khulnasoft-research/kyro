#![allow(dead_code)]

use crate::scheduler::block_manager::BlockId;
use candle_core::{Result, Tensor};

pub struct KVCache {
    blocks: Vec<Vec<u32>>,
    block_size: usize,
}

impl KVCache {
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            block_size,
        }
    }

    pub fn update(
        &mut self,
        block_id: BlockId,
        _slot_idx: usize,
        _key: &Tensor,
        _value: &Tensor,
    ) -> Result<()> {
        let block_idx = block_id.0;
        while self.blocks.len() <= block_idx {
            self.blocks.push(Vec::new());
        }
        Ok(())
    }
}
