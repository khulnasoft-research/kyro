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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_returns_single_node() {
        let ctx = DistributedContext::new();
        assert_eq!(ctx.rank, 0);
        assert_eq!(ctx.world_size, 1);
    }

    #[test]
    fn test_default_equals_new() {
        let from_new = DistributedContext::new();
        let from_default = DistributedContext::default();
        assert_eq!(from_new.rank, from_default.rank);
        assert_eq!(from_new.world_size, from_default.world_size);
    }
}
