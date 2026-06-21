pub struct DistributedContext {
    pub rank: u32,
    pub world_size: u32,
}

#[allow(dead_code)]
impl DistributedContext {
    pub fn new() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }
}

impl Default for DistributedContext {
    fn default() -> Self {
        Self::new()
    }
}
