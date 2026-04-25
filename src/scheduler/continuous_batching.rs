use crate::scheduler::block_manager::BlockManager;
use std::collections::VecDeque;

pub struct Request {
    pub id: u64,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub is_prefill: bool,
    /// Number of prompt tokens already covered by a Radix Cache hit.
    /// The model only needs to process `prompt_tokens[cached_prefix_len..]`.
    pub cached_prefix_len: usize,
    /// For Chunked Prefill: the index into prompt_tokens up to which we've prefilled so far.
    pub prefill_cursor: usize,
    pub temperature: f32,
    pub top_p: f32,
    /// Channel to send newly generated tokens back to the API for streaming.
    pub token_sender: Option<tokio::sync::mpsc::UnboundedSender<u32>>,
    /// Optional grammar processor for structured output (XGrammar).
    pub grammar_processor: Option<crate::api::grammar::GrammarLogitsProcessor>,
}

pub const PREFILL_CHUNK_SIZE: usize = 512;

pub struct SchedulerConfig {
    pub max_tokens_per_iter: usize,
    pub max_prefill_chunk_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_iter: 2048,
            max_prefill_chunk_size: 512,
        }
    }
}

pub struct Scheduler {
    pub waiting_queue: VecDeque<Request>,
    pub running_queue: Vec<Request>,
    pub block_manager: BlockManager,
    pub config: SchedulerConfig,
}

impl Scheduler {
    pub fn new(block_manager: BlockManager, config: SchedulerConfig) -> Self {
        Self {
            waiting_queue: VecDeque::new(),
            running_queue: Vec::new(),
            block_manager,
            config,
        }
    }

    pub fn add_request(&mut self, request: Request) {
        self.waiting_queue.push_back(request);
    }

    pub fn schedule(&mut self) -> (Vec<u64>, Vec<u64>) {
        let mut to_prefill = Vec::new();
        let mut to_decode = Vec::new();
        let mut total_tokens = 0;

        // 1. Prioritize ongoing Decodes (Memory Bound)
        for req in &mut self.running_queue {
            if req.prefill_cursor >= req.prompt_tokens.len() {
                to_decode.push(req.id);
                req.is_prefill = false;
                total_tokens += 1;
            }
        }

        // 2. Add Prefills (Compute Bound), but chunked
        // First, check already running prefill requests
        for req in &mut self.running_queue {
            if total_tokens >= self.config.max_tokens_per_iter {
                break;
            }

            if req.prefill_cursor < req.prompt_tokens.len() {
                let remaining = req.prompt_tokens.len() - req.prefill_cursor;
                let chunk_size = std::cmp::min(remaining, self.config.max_prefill_chunk_size);

                req.is_prefill = true;
                to_prefill.push(req.id);
                total_tokens += chunk_size;
            }
        }

        // Second, pull new requests from waiting queue
        while let Some(req) = self.waiting_queue.front() {
            if total_tokens >= self.config.max_tokens_per_iter {
                break;
            }

            let tokens = req.prompt_tokens.clone();
            if let Some((_blocks, cached_len)) =
                self.block_manager.allocate_with_prefix(req.id, &tokens)
            {
                let mut req = self.waiting_queue.pop_front().unwrap();
                req.cached_prefix_len = cached_len;
                req.prefill_cursor = cached_len;

                let remaining = req.prompt_tokens.len() - req.prefill_cursor;
                let chunk_size = std::cmp::min(remaining, self.config.max_prefill_chunk_size);

                if remaining > 0 {
                    req.is_prefill = true;
                    to_prefill.push(req.id);
                    total_tokens += chunk_size;
                } else {
                    // Fully cached, move to decode
                    req.is_prefill = false;
                    to_decode.push(req.id);
                    total_tokens += 1;
                }

                self.running_queue.push(req);
            } else {
                break;
            }
        }

        (to_prefill, to_decode)
    }

    pub fn advance_prefill_cursor(&mut self, request_id: u64) {
        if let Some(req) = self.running_queue.iter_mut().find(|r| r.id == request_id) {
            let next = (req.prefill_cursor + self.config.max_prefill_chunk_size)
                .min(req.prompt_tokens.len());
            req.prefill_cursor = next;
        }
    }

    pub fn finish_request(&mut self, request_id: u64) {
        if let Some(pos) = self.running_queue.iter().position(|r| r.id == request_id) {
            self.running_queue.remove(pos);
            self.block_manager.free(request_id);
        }
    }
}
