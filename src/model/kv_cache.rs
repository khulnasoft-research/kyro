use crate::scheduler::block_manager::BlockId;
use candle_core::{Result, Tensor};

#[allow(dead_code)]
pub struct KVCache {
    blocks: Vec<Vec<u32>>,
    block_size: usize,
}

impl KVCache {
    #[allow(dead_code)]
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            block_size,
        }
    }

    #[allow(dead_code)]
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
