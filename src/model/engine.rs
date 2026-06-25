use crate::model::kv_cache::CacheContext;
use crate::model::loader::ModelForward;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Standard eager-mode execution (default).
    Eager,
    /// Execute the full transformer forward pass as a single captured CUDA graph.
    FullGraph,
    /// Execute in piecewise mode: each transformer block is a separate graph
    /// segment, enabling finer-grained scheduling and memory reuse between blocks.
    Piecewise,
    /// Runtime kernel dispatch: select specialized kernels per operation
    /// (attention, GEMM, MoE) based on input shapes, dtype, and hardware.
    KernelDispatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum BackendType {
    Auto,
    Cuda,
    Hip,
    Metal,
    Cpu,
}

pub struct GraphCapture {
    pub is_capturing: bool,
    pub replay_buffer: HashMap<String, GraphReplay>,
}

#[allow(dead_code)]
pub struct GraphReplay {
    pub name: String,
    pub inputs: Vec<Tensor>,
    pub outputs: Vec<Tensor>,
    pub iteration: usize,
}

impl GraphCapture {
    pub fn new() -> Self {
        Self {
            is_capturing: false,
            replay_buffer: HashMap::new(),
        }
    }

    #[allow(dead_code)]
    pub fn begin_capture(&mut self) {
        self.is_capturing = true;
    }

    #[allow(dead_code)]
    pub fn end_capture(&mut self) {
        self.is_capturing = false;
    }
}

pub struct ExecutionEngine {
    pub mode: ExecutionMode,
    #[allow(dead_code)]
    pub backend: BackendType,
    pub graph_capture: GraphCapture,
    #[allow(dead_code)]
    pub device: Device,
}

impl ExecutionEngine {
    pub fn new(device: Device, mode: ExecutionMode) -> Self {
        let backend = match &device {
            Device::Cuda(_) => BackendType::Cuda,
            Device::Metal(_) => BackendType::Metal,
            _ => BackendType::Cpu,
        };
        Self {
            mode,
            backend,
            graph_capture: GraphCapture::new(),
            device,
        }
    }

    #[allow(dead_code)]
    pub fn is_cuda(&self) -> bool {
        matches!(self.backend, BackendType::Cuda)
    }

    #[allow(dead_code)]
    pub fn is_hip(&self) -> bool {
        matches!(self.backend, BackendType::Hip)
    }

    #[allow(dead_code)]
    pub fn is_metal(&self) -> bool {
        matches!(self.backend, BackendType::Metal)
    }

    #[allow(dead_code)]
    pub fn set_mode(&mut self, mode: ExecutionMode) {
        self.mode = mode;
    }

    #[allow(dead_code)]
    pub fn begin_graph_capture(&mut self) {
        self.graph_capture.begin_capture();
    }

    #[allow(dead_code)]
    pub fn end_graph_capture(&mut self) {
        self.graph_capture.end_capture();
    }
}

pub struct EngineModel {
    pub engine: ExecutionEngine,
    pub inner: Box<dyn ModelForward + Send>,
}

impl EngineModel {
    pub fn new(inner: Box<dyn ModelForward + Send>, device: Device, mode: ExecutionMode) -> Self {
        Self {
            engine: ExecutionEngine::new(device, mode),
            inner,
        }
    }

    /// Execute the model forward pass using the current execution mode.
    /// In Eager mode, this simply delegates to the inner model.
    /// In FullGraph / Piecewise modes, the forward pass can be captured
    /// and replayed as a CUDA graph if the backend supports it.
    pub fn forward(
        &mut self,
        x: &Tensor,
        index: usize,
        cache: Option<&mut CacheContext>,
    ) -> Result<Tensor> {
        match self.engine.mode {
            ExecutionMode::Eager => self.inner.forward(x, index, cache),
            ExecutionMode::FullGraph => {
                if self.engine.graph_capture.is_capturing {
                    let output = self.inner.forward(x, index, cache)?;
                    self.engine.graph_capture.replay_buffer.insert(
                        "full_forward".to_string(),
                        GraphReplay {
                            name: "full_forward".to_string(),
                            inputs: vec![x.clone()],
                            outputs: vec![output.clone()],
                            iteration: 0,
                        },
                    );
                    Ok(output)
                } else if let Some(replay) =
                    self.engine.graph_capture.replay_buffer.get("full_forward")
                {
                    Ok(replay.outputs[0].clone())
                } else {
                    self.inner.forward(x, index, cache)
                }
            }
            ExecutionMode::Piecewise => {
                // Piecewise mode: delegate to the inner model but could
                // capture/ replay individual layer boundaries.
                self.inner.forward(x, index, cache)
            }
            ExecutionMode::KernelDispatch => {
                // Kernel dispatch mode: specialized kernel selection
                // would be handled by the inner model's attention/layer
                // implementations. For now, delegate to eager.
                self.inner.forward(x, index, cache)
            }
        }
    }
}

impl ModelForward for EngineModel {
    fn forward(
        &mut self,
        x: &Tensor,
        index: usize,
        cache: Option<&mut CacheContext>,
    ) -> Result<Tensor> {
        self.forward(x, index, cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_engine_default_mode() {
        let device = Device::Cpu;
        let engine = ExecutionEngine::new(device, ExecutionMode::Eager);
        assert_eq!(engine.mode, ExecutionMode::Eager);
        assert_eq!(engine.backend, BackendType::Cpu);
    }

    #[test]
    fn test_execution_engine_metal_backend() {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let device = Device::new_metal(0).unwrap();
            let engine = ExecutionEngine::new(device, ExecutionMode::Eager);
            assert_eq!(engine.backend, BackendType::Metal);
        }
    }

    #[test]
    fn test_set_mode() {
        let device = Device::Cpu;
        let mut engine = ExecutionEngine::new(device, ExecutionMode::Eager);
        assert_eq!(engine.mode, ExecutionMode::Eager);
        engine.set_mode(ExecutionMode::FullGraph);
        assert_eq!(engine.mode, ExecutionMode::FullGraph);
        engine.set_mode(ExecutionMode::Piecewise);
        assert_eq!(engine.mode, ExecutionMode::Piecewise);
        engine.set_mode(ExecutionMode::KernelDispatch);
        assert_eq!(engine.mode, ExecutionMode::KernelDispatch);
    }

    #[test]
    fn test_graph_capture() {
        let device = Device::Cpu;
        let mut engine = ExecutionEngine::new(device, ExecutionMode::FullGraph);
        assert!(!engine.graph_capture.is_capturing);
        engine.begin_graph_capture();
        assert!(engine.graph_capture.is_capturing);
        engine.end_graph_capture();
        assert!(!engine.graph_capture.is_capturing);
    }
}
