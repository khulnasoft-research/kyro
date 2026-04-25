use crate::scheduler::block_manager::BlockId;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug)]
struct RadixNode {
    /// The sequence of tokens stored in this node
    tokens: Vec<u32>,
    /// The physical block IDs corresponding to these tokens
    block_ids: Vec<BlockId>,
    /// Children nodes mapped by the first token of the next sub-sequence
    children: HashMap<u32, Box<RadixNode>>,
    /// Last time this prefix was accessed (for LRU eviction)
    last_accessed: Instant,
}

impl RadixNode {
    fn new(tokens: Vec<u32>, block_ids: Vec<BlockId>) -> Self {
        Self {
            tokens,
            block_ids,
            children: HashMap::new(),
            last_accessed: Instant::now(),
        }
    }
}

pub struct RadixCache {
    root: RadixNode,
    /// Total number of blocks currently cached
    pub num_cached_blocks: usize,
    /// Maximum blocks allowed in cache before eviction
    pub max_capacity: usize,
}

impl RadixCache {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            root: RadixNode::new(vec![], vec![]),
            num_cached_blocks: 0,
            max_capacity,
        }
    }

    /// Matches a sequence of tokens against the cache.
    /// Returns (Cached Block IDs, tokens_matched_count)
    pub fn match_prefix(&mut self, tokens: &[u32]) -> (Vec<BlockId>, usize) {
        let mut current_node = &mut self.root;
        let mut matched_blocks = Vec::new();
        let mut total_matched_tokens = 0;

        let mut token_idx = 0;
        while token_idx < tokens.len() {
            let first_token = tokens[token_idx];

            if let Some(child) = current_node.children.get_mut(&first_token) {
                // Check if the rest of the child's tokens match
                let match_len = child
                    .tokens
                    .iter()
                    .zip(&tokens[token_idx..])
                    .take_while(|(a, b)| a == b)
                    .count();

                if match_len == child.tokens.len() {
                    // Full node match, move deeper
                    matched_blocks.extend_from_slice(&child.block_ids);
                    total_matched_tokens += match_len;
                    token_idx += match_len;
                    child.last_accessed = Instant::now();
                    current_node = child;
                } else {
                    // Partial match (prefix of a node)
                    break;
                }
            } else {
                break;
            }
        }

        (matched_blocks, total_matched_tokens)
    }

    /// Inserts a new sequence of tokens and their computed blocks into the cache
    pub fn insert(&mut self, tokens: &[u32], block_ids: &[BlockId]) {
        let mut current_node = &mut self.root;
        let mut token_idx = 0;

        while token_idx < tokens.len() {
            let first_token = tokens[token_idx];

            if !current_node.children.contains_key(&first_token) {
                let new_node = RadixNode::new(tokens[token_idx..].to_vec(), block_ids.to_vec());
                current_node
                    .children
                    .insert(first_token, Box::new(new_node));
                self.num_cached_blocks += block_ids.len();
                break;
            }

            let child = current_node.children.get_mut(&first_token).unwrap();
            token_idx += child.tokens.len();
            current_node = child;
        }

        if self.num_cached_blocks > self.max_capacity {
            self.evict_lru();
        }
    }

    pub fn evict_lru(&mut self) -> Vec<BlockId> {
        let mut blocks_to_free = Vec::new();
        loop {
            let blocks = self.remove_oldest_leaf();
            match blocks {
                Some(evicted_blocks) => {
                    self.num_cached_blocks -= evicted_blocks.len();
                    blocks_to_free.extend(evicted_blocks);
                    if self.num_cached_blocks <= self.max_capacity {
                        break;
                    }
                }
                None => break,
            }
        }
        blocks_to_free
    }

    fn remove_oldest_leaf(&mut self) -> Option<Vec<BlockId>> {
        let node = &mut self.root;
        if node.children.is_empty() {
            return None;
        }

        let mut oldest_token = None;
        let mut oldest_time = Instant::now();

        for (token, child) in &node.children {
            if child.children.is_empty() {
                if child.last_accessed < oldest_time {
                    oldest_time = child.last_accessed;
                    oldest_token = Some(*token);
                }
            }
        }

        if let Some(token) = oldest_token {
            let child = node.children.remove(&token).unwrap();
            Some(child.block_ids)
        } else {
            None
        }
    }
}
