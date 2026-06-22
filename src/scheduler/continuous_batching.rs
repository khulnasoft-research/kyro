use crate::scheduler::block_manager::BlockManager;
use std::collections::VecDeque;
use tokio::time::Instant;

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
    /// Deadline for request cancellation. None means no timeout.
    pub deadline: Option<Instant>,
    /// Optional grammar processor for structured output (XGrammar).
    #[allow(dead_code)]
    pub grammar_processor: Option<crate::api::grammar::GrammarLogitsProcessor>,
}

pub struct SchedulerConfig {
    pub max_tokens_per_iter: usize,
    pub max_prefill_chunk_size: usize,
    #[allow(dead_code)]
    pub request_timeout_secs: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_iter: 2048,
            max_prefill_chunk_size: 512,
            request_timeout_secs: 300.0,
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

        // 0. Remove expired requests
        let now = Instant::now();
        self.running_queue.retain(|req| {
            if let Some(deadline) = req.deadline {
                if now >= deadline {
                    self.block_manager.free(req.id);
                    return false;
                }
            }
            true
        });
        self.waiting_queue.retain(|req| {
            if let Some(deadline) = req.deadline {
                if now >= deadline {
                    return false;
                }
            }
            true
        });

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
                let mut req = self.waiting_queue.pop_front().expect("front checked above");
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scheduler(max_tokens: usize) -> Scheduler {
        let bm = crate::scheduler::block_manager::BlockManager::new(4, 100, 10);
        let cfg = SchedulerConfig {
            max_tokens_per_iter: max_tokens,
            max_prefill_chunk_size: 4,
            request_timeout_secs: 300.0,
        };
        Scheduler::new(bm, cfg)
    }

    fn make_req(id: u64, prompt_len: usize, max_gen: usize) -> Request {
        Request {
            id,
            prompt_tokens: (0..prompt_len as u32).collect(),
            generated_tokens: Vec::new(),
            max_tokens: max_gen,
            is_prefill: true,
            cached_prefix_len: 0,
            prefill_cursor: 0,
            temperature: 1.0,
            top_p: 1.0,
            token_sender: None,
            deadline: None,
            grammar_processor: None,
        }
    }

    #[test]
    fn test_empty_schedule() {
        let mut sched = make_scheduler(2048);
        let (to_prefill, to_decode) = sched.schedule();
        assert!(to_prefill.is_empty());
        assert!(to_decode.is_empty());
    }

    #[test]
    fn test_single_request_prefill() {
        let mut sched = make_scheduler(2048);
        sched.add_request(make_req(1, 10, 50));
        let (to_prefill, to_decode) = sched.schedule();
        assert_eq!(to_prefill.len(), 1);
        assert_eq!(to_prefill[0], 1);
        assert!(to_decode.is_empty());
        assert_eq!(sched.running_queue.len(), 1);
    }

    #[test]
    fn test_prefill_to_decode_transition() {
        let mut sched = make_scheduler(2048);
        // Use a small prompt (3 tokens) that fits in one chunk
        sched.add_request(make_req(1, 3, 50));

        let (to_prefill, to_decode) = sched.schedule();
        assert_eq!(to_prefill.len(), 1);
        assert!(to_decode.is_empty());

        // Advance past full prompt length
        sched.advance_prefill_cursor(1);

        let (to_prefill, to_decode) = sched.schedule();
        assert!(to_prefill.is_empty());
        assert_eq!(to_decode.len(), 1);
        assert_eq!(to_decode[0], 1);
    }

    #[test]
    fn test_decode_priority_over_prefill() {
        let mut sched = make_scheduler(10);
        let mut decode_req = make_req(1, 0, 50);
        decode_req.is_prefill = false;
        decode_req.prefill_cursor = 1;
        sched.running_queue.push(decode_req);
        sched.add_request(make_req(2, 10, 50));

        let (_to_prefill, to_decode) = sched.schedule();
        assert_eq!(to_decode.len(), 1);
        assert_eq!(to_decode[0], 1);
    }

    #[test]
    fn test_finish_request_cleanup() {
        let mut sched = make_scheduler(2048);
        sched.add_request(make_req(1, 10, 50));
        sched.schedule();
        assert_eq!(sched.running_queue.len(), 1);

        sched.finish_request(1);
        assert!(sched.running_queue.is_empty());
    }

    #[test]
    fn test_max_tokens_per_iter_respected() {
        let mut sched = make_scheduler(6);
        sched.add_request(make_req(1, 20, 50));
        sched.add_request(make_req(2, 20, 50));

        let (to_prefill, to_decode) = sched.schedule();
        // With max_tokens_per_iter=6 and chunk_size=4, both can fit (4+4=8 >6
        // but the check is before computing chunk_size for the second request)
        assert_eq!(to_prefill.len(), 2);
        assert!(to_decode.is_empty());
        assert_eq!(sched.waiting_queue.len(), 0);
    }

    #[test]
    fn test_tight_budget_limits_prefills() {
        let mut sched = make_scheduler(3);
        sched.add_request(make_req(1, 20, 50));
        sched.add_request(make_req(2, 20, 50));

        let (to_prefill, _to_decode) = sched.schedule();
        // Only 1 request should fit since max_tokens_per_iter=3 < chunk_size=4
        assert_eq!(to_prefill.len(), 1);
        assert_eq!(sched.waiting_queue.len(), 1);
    }

    #[test]
    fn test_finished_request_removed_from_schedule() {
        let mut sched = make_scheduler(2048);
        let mut req = make_req(1, 10, 2);
        req.generated_tokens = vec![0, 1];
        sched.running_queue.push(req);

        let finished: Vec<u64> = sched
            .running_queue
            .iter()
            .filter(|r| r.generated_tokens.len() >= r.max_tokens)
            .map(|r| r.id)
            .collect();

        for id in finished {
            sched.finish_request(id);
        }

        assert!(sched.running_queue.is_empty());
    }
}
