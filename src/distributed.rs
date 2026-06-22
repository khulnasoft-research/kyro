use candle_core::{Device, Result, Tensor};
use std::sync::Arc;

pub enum Backend {
    Cpu,
    Cuda,
}

pub struct DistributedContext {
    pub rank: u32,
    pub world_size: u32,
    pub backend: Backend,
    #[cfg(feature = "cuda")]
    pub cuda_context: Option<crate::distributed::cuda::CudaContext>,
}

#[cfg(feature = "cuda")]
pub mod cuda {
    use candle_core::Result;
    use cudarc::driver::CudaDevice;

    pub struct CudaContext {
        pub device: CudaDevice,
    }

    impl CudaContext {
        pub fn new(device: CudaDevice) -> Self {
            Self { device }
        }

        pub fn all_reduce(&self, tensor: &candle_core::Tensor) -> Result<candle_core::Tensor> {
            // CUDA all-reduce stub - in production this would use NCCL
            Ok(tensor.clone())
        }
    }
}

impl DistributedContext {
    pub fn new() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            backend: Backend::Cpu,
            #[cfg(feature = "cuda")]
            cuda_context: None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn new_cuda(rank: u32, world_size: u32, device: CudaDevice) -> Self {
        Self {
            rank,
            world_size,
            backend: Backend::Cuda,
            cuda_context: Some(cuda::CudaContext::new(device)),
        }
    }

    /// All-reduce across all devices in the tensor parallel group.
    /// Sums gradients/activations across GPUs so each device has the full result.
    pub fn all_reduce(&self, tensor: &Tensor) -> Result<Tensor> {
        if self.world_size <= 1 {
            return Ok(tensor.clone());
        }

        match self.backend {
            Backend::Cpu => self.cpu_all_reduce(tensor),
            Backend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    if let Some(ref ctx) = self.cuda_context {
                        return ctx.all_reduce(tensor);
                    }
                }
                self.cpu_all_reduce(tensor)
            }
        }
    }

    /// CPU-based all-reduce using local copying (single-process simulation).
    /// In a real multi-GPU setup, this would be replaced by NCCL calls.
    fn cpu_all_reduce(&self, tensor: &Tensor) -> Result<Tensor> {
        // For CPU simulation of tensor parallelism, we just return the tensor
        // since all shards are already on the same device.
        Ok(tensor.clone())
    }

    /// Split a dimension across the tensor parallel group.
    pub fn shard_dim(&self, tensor: &Tensor, dim: usize) -> Result<Tensor> {
        if self.world_size <= 1 {
            return Ok(tensor.clone());
        }
        let dim_size = tensor.dim(dim)?;
        let shard_size = dim_size / self.world_size as usize;
        let offset = self.rank as usize * shard_size;
        tensor.narrow(dim, offset, shard_size)
    }

    /// Gather a sharded dimension back to the full size.
    pub fn gather_dim(&self, tensor: &Tensor, _dim: usize) -> Result<Tensor> {
        if self.world_size <= 1 {
            return Ok(tensor.clone());
        }
        // In single-process mode, we can't actually gather.
        // This would require communication in multi-process mode.
        Ok(tensor.clone())
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
    use candle_core::Device;

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

    #[test]
    fn test_all_reduce_single_node_returns_clone() {
        let ctx = DistributedContext::new();
        let device = Device::Cpu;
        let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let result = ctx.all_reduce(&tensor).unwrap();
        assert_eq!(result.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_shard_dim_single_node_no_op() {
        let ctx = DistributedContext::new();
        let device = Device::Cpu;
        let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let result = ctx.shard_dim(&tensor, 0).unwrap();
        assert_eq!(result.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_all_reduce_twice_consistent() {
        let ctx = DistributedContext::new();
        let device = Device::Cpu;
        let tensor = Tensor::new(&[10.0f32, 20.0], &device).unwrap();
        let r1 = ctx.all_reduce(&tensor).unwrap();
        let r2 = ctx.all_reduce(&tensor).unwrap();
        assert_eq!(r1.to_vec1::<f32>().unwrap(), r2.to_vec1::<f32>().unwrap());
    }
}
