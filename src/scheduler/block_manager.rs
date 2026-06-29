use std::collections::HashMap;

use crate::scheduler::radix_cache::RadixCache;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

#[allow(dead_code)]
pub struct PhysicalBlock {
    pub id: BlockId,
    pub device_id: usize,
    pub ref_count: usize,
}

pub struct BlockManager {
    pub block_size: usize,
    #[allow(dead_code)]
    pub num_gpu_blocks: usize,
    #[allow(dead_code)]
    pub num_cpu_blocks: usize,
    pub free_gpu_blocks: Vec<BlockId>,
    #[allow(dead_code)]
    pub free_cpu_blocks: Vec<BlockId>,
    pub block_table: HashMap<u64, Vec<BlockId>>, // request_id -> block_ids
    pub radix_cache: RadixCache,
    // Tracks the prompt token sequence for each request (needed for cache insertion on free)
    pub prompt_table: HashMap<u64, Vec<u32>>,
    /// Tracks the reference count of each physical block
    pub ref_counts: Vec<usize>,
}

impl BlockManager {
    pub fn new(block_size: usize, num_gpu_blocks: usize, num_cpu_blocks: usize) -> Self {
        // Reserve 30% of GPU blocks as the Radix Cache's eviction budget
        let cache_capacity = num_gpu_blocks / 3;
        let free_gpu_blocks = (0..num_gpu_blocks).map(BlockId).collect();
        let free_cpu_blocks = (0..num_cpu_blocks).map(BlockId).collect();
        Self {
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            free_gpu_blocks,
            free_cpu_blocks,
            block_table: HashMap::new(),
            radix_cache: RadixCache::new(cache_capacity),
            prompt_table: HashMap::new(),
            ref_counts: vec![0; num_gpu_blocks],
        }
    }

    /// Allocates blocks for a request, first checking the Radix Cache for any
    /// prefix hit to avoid recomputation. Returns (allocated_blocks, cached_token_count).
    pub fn allocate_with_prefix(
        &mut self,
        request_id: u64,
        prompt_tokens: &[u32],
    ) -> Option<(Vec<BlockId>, usize)> {
        // 1. Check Radix Cache for prefix hit
        let (cached_blocks, cached_token_count) = self.radix_cache.match_prefix(prompt_tokens);

        // Increment ref count for cached blocks
        for block in &cached_blocks {
            self.ref_counts[block.0] += 1;
        }

        // 2. Calculate how many NEW blocks we need (beyond the cache hit)
        let remaining_tokens = prompt_tokens.len().saturating_sub(cached_token_count);
        let new_blocks_needed = remaining_tokens.div_ceil(self.block_size);

        // 3. Check we have enough free GPU blocks for the remainder
        // First try to evict from the radix cache if needed
        if self.free_gpu_blocks.len() < new_blocks_needed {
            let evicted = self.radix_cache.evict_lru();
            for block in evicted {
                self.ref_counts[block.0] -= 1;
                if self.ref_counts[block.0] == 0 {
                    self.free_gpu_blocks.push(block);
                }
            }
        }

        if self.free_gpu_blocks.len() < new_blocks_needed {
            // Rollback ref counts on failure
            for block in &cached_blocks {
                self.ref_counts[block.0] -= 1;
            }
            return None; // Out of memory even after eviction
        }

        // 4. Allocate new blocks for the uncached suffix
        let mut all_blocks = cached_blocks;
        for _ in 0..new_blocks_needed {
            let block = self
                .free_gpu_blocks
                .pop()
                .expect("free blocks checked above");
            self.ref_counts[block.0] = 1; // 1 for the request
            all_blocks.push(block);
        }

        self.block_table.insert(request_id, all_blocks.clone());
        self.prompt_table.insert(request_id, prompt_tokens.to_vec());

        Some((all_blocks, cached_token_count))
    }

    /// Legacy allocate (no prefix caching). Kept for compatibility.
    #[allow(dead_code)]
    pub fn allocate(&mut self, request_id: u64, num_tokens: usize) -> Option<Vec<BlockId>> {
        let num_blocks = num_tokens.div_ceil(self.block_size);
        if self.free_gpu_blocks.len() < num_blocks {
            return None;
        }
        let mut allocated = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let block = self
                .free_gpu_blocks
                .pop()
                .expect("free blocks checked above");
            self.ref_counts[block.0] = 1;
            allocated.push(block);
        }
        self.block_table.insert(request_id, allocated.clone());
        Some(allocated)
    }

    /// Frees a request's blocks back into the Radix Cache (not the free pool directly).
    /// The LRU eviction policy manages when blocks are truly released.
    pub fn free(&mut self, request_id: u64) {
        if let Some(blocks) = self.block_table.remove(&request_id) {
            let tokens = self.prompt_table.remove(&request_id);

            for block in &blocks {
                self.ref_counts[block.0] -= 1;
            }

            if let Some(tokens) = tokens {
                // Deposit blocks into the Radix Cache for future prefix reuse
                // We increment ref counts because the cache now 'owns' a reference
                for block in &blocks {
                    self.ref_counts[block.0] += 1;
                }
                self.radix_cache.insert(&tokens, &blocks);
            } else {
                // If no token trace, release blocks with ref count 0 to free pool
                for block in blocks {
                    if self.ref_counts[block.0] == 0 {
                        self.free_gpu_blocks.push(block);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager(gpu_blocks: usize) -> BlockManager {
        BlockManager::new(4, gpu_blocks, 2)
    }

    #[test]
    fn test_allocate_with_prefix_no_cache() {
        let mut bm = make_manager(10);
        let tokens = vec![1, 2, 3, 4, 5];
        let result = bm.allocate_with_prefix(1, &tokens);
        assert!(result.is_some());
        let (blocks, cached) = result.unwrap();
        // 5 tokens, block_size=4 -> 2 blocks needed
        assert_eq!(blocks.len(), 2);
        assert_eq!(cached, 0);
        assert_eq!(bm.free_gpu_blocks.len(), 10 - 2);
    }

    #[test]
    fn test_allocate_with_prefix_full_cache_hit() {
        let mut bm = make_manager(10);
        let tokens = vec![1, 2, 3, 4];
        // First request populates cache
        bm.allocate_with_prefix(1, &tokens);
        bm.free(1);
        // Second request should hit cache
        let result = bm.allocate_with_prefix(2, &tokens);
        assert!(result.is_some());
        let (blocks, cached) = result.unwrap();
        assert!(!blocks.is_empty());
        assert_eq!(cached, 4);
    }

    #[test]
    fn test_allocate_with_prefix_partial_cache_hit() {
        let mut bm = make_manager(10);
        let prefix = vec![1, 2];
        let full = vec![1, 2, 3, 4, 5];
        // Cache the prefix
        bm.allocate_with_prefix(1, &prefix);
        bm.free(1);
        // Now request full sequence - should hit cache for first 2 tokens
        let result = bm.allocate_with_prefix(2, &full);
        assert!(result.is_some());
        let (_blocks, cached) = result.unwrap();
        assert_eq!(cached, 2);
    }

    #[test]
    fn test_oom_returns_none() {
        let mut bm = make_manager(1);
        let tokens = vec![1, 2, 3, 4, 5];
        let result = bm.allocate_with_prefix(1, &tokens);
        assert!(result.is_none());
    }

    #[test]
    fn test_free_reuses_blocks() {
        let mut bm = make_manager(10);
        let tokens = vec![1, 2, 3, 4];
        bm.allocate_with_prefix(1, &tokens);
        let free_before = bm.free_gpu_blocks.len();
        bm.free(1);
        // After free, blocks go to cache, not free pool
        assert_eq!(bm.free_gpu_blocks.len(), free_before);
        assert!(bm.radix_cache.num_cached_blocks > 0);
    }

    #[test]
    fn test_multiple_requests_unique_block_tables() {
        let mut bm = make_manager(20);
        bm.allocate_with_prefix(1, &[1, 2, 3, 4]);
        bm.allocate_with_prefix(2, &[5, 6, 7, 8]);
        assert!(bm.block_table.contains_key(&1));
        assert!(bm.block_table.contains_key(&2));
        assert_ne!(bm.block_table[&1], bm.block_table[&2]);
    }
}
