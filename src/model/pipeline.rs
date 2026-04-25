pub struct PipelineContext {
    pub rank: usize,
    pub world_size: usize,
    pub start_layer: usize,
    pub end_layer: usize,
}

impl PipelineContext {
    pub fn new(rank: usize, world_size: usize, total_layers: usize) -> Self {
        let layers_per_gpu = (total_layers + world_size - 1) / world_size;
        let start_layer = rank * layers_per_gpu;
        let end_layer = std::cmp::min(start_layer + layers_per_gpu, total_layers);

        Self {
            rank,
            world_size,
            start_layer,
            end_layer,
        }
    }

    pub fn is_first_stage(&self) -> bool {
        self.rank == 0
    }

    pub fn is_last_stage(&self) -> bool {
        self.rank == self.world_size - 1
    }

    pub fn should_process_layer(&self, index: usize) -> bool {
        index >= self.start_layer && index < self.end_layer
    }
}
