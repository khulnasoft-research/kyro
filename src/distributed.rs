use std::sync::atomic::{AtomicU32, Ordering};

pub struct DistributedContext {
    pub rank: u32,
    pub world_size: u32,
}

impl DistributedContext {
    pub fn new() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }

    pub fn is_main_process(&self) -> bool {
        self.rank == 0
    }

    pub fn all_reduce(&self, _data: &[f32]) -> Vec<f32> {
        vec![]
    }
}

impl Default for DistributedContext {
    fn default() -> Self {
        Self::new()
    }
}
