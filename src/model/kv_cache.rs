use candle_core::{Result, Tensor};
use std::collections::HashMap;

pub struct RequestKVCache {
    pub key_tensor: Option<Tensor>,
    pub value_tensor: Option<Tensor>,
    pub num_cached_tokens: usize,
}

impl RequestKVCache {
    pub fn new() -> Self {
        Self {
            key_tensor: None,
            value_tensor: None,
            num_cached_tokens: 0,
        }
    }
}

pub struct KVCacheManager {
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub request_caches: HashMap<u64, RequestKVCache>,
}

impl KVCacheManager {
    pub fn new(num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_kv_heads,
            head_dim,
            request_caches: HashMap::new(),
        }
    }

    pub fn register_request(&mut self, request_id: u64) {
        self.request_caches.insert(request_id, RequestKVCache::new());
    }

    pub fn unregister_request(&mut self, request_id: u64) {
        self.request_caches.remove(&request_id);
    }

    /// Append K and V tensors for a request.
    /// Both key and value should be [batch, seq_len, num_kv_heads, head_dim].
    pub fn append_kv(
        &mut self,
        request_id: u64,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<()> {
        let req_cache = self
            .request_caches
            .get_mut(&request_id)
            .expect("request must be registered");

        let k_f16 = key.to_dtype(candle_core::DType::F16)?;
        let v_f16 = value.to_dtype(candle_core::DType::F16)?;

        req_cache.key_tensor = match req_cache.key_tensor.take() {
            Some(existing) => {
                let seq_dim = 1;
                Some(Tensor::cat(&[&existing, &k_f16], seq_dim)?)
            }
            None => Some(k_f16),
        };
        req_cache.value_tensor = match req_cache.value_tensor.take() {
            Some(existing) => {
                let seq_dim = 1;
                Some(Tensor::cat(&[&existing, &v_f16], seq_dim)?)
            }
            None => Some(v_f16),
        };

        req_cache.num_cached_tokens += key.dim(1)?;
        Ok(())
    }

    pub fn get_context_len(&self, request_id: u64) -> usize {
        self.request_caches
            .get(&request_id)
            .map(|c| c.num_cached_tokens)
            .unwrap_or(0)
    }

    pub fn get_cached_key(&self, request_id: u64) -> Option<Tensor> {
        self.request_caches
            .get(&request_id)
            .and_then(|c| c.key_tensor.clone())
    }

    pub fn get_cached_value(&self, request_id: u64) -> Option<Tensor> {
        self.request_caches
            .get(&request_id)
            .and_then(|c| c.value_tensor.clone())
    }
}

/// Wrapper for passing cache context through the model forward chain.
/// Avoids issues with moving &mut references in closures.
pub struct CacheContext<'a> {
    pub manager: &'a mut KVCacheManager,
    pub request_id: u64,
}

impl<'a> CacheContext<'a> {
    pub fn new(manager: &'a mut KVCacheManager, request_id: u64) -> Self {
        Self { manager, request_id }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_kv_cache_create_and_register() {
        let mut manager = KVCacheManager::new(4, 64);
        manager.register_request(1);
        assert_eq!(manager.get_context_len(1), 0);
        manager.unregister_request(1);
        assert!(manager.request_caches.is_empty());
    }

    #[test]
    fn test_append_kv_increases_context() {
        let device = Device::Cpu;
        let mut manager = KVCacheManager::new(4, 64);
        manager.register_request(42);

        let key = Tensor::zeros((1, 2, 4, 64), candle_core::DType::F16, &device).unwrap();
        let value = Tensor::zeros((1, 2, 4, 64), candle_core::DType::F16, &device).unwrap();
        manager.append_kv(42, &key, &value).unwrap();
        assert_eq!(manager.get_context_len(42), 2);

        // Append more
        let key2 = Tensor::zeros((1, 3, 4, 64), candle_core::DType::F16, &device).unwrap();
        let value2 = Tensor::zeros((1, 3, 4, 64), candle_core::DType::F16, &device).unwrap();
        manager.append_kv(42, &key2, &value2).unwrap();
        assert_eq!(manager.get_context_len(42), 5);
    }

    #[test]
    fn test_get_cached_tensors() {
        let device = Device::Cpu;
        let mut manager = KVCacheManager::new(4, 64);
        manager.register_request(7);

        let key = Tensor::ones((1, 1, 4, 64), candle_core::DType::F16, &device).unwrap();
        let value = Tensor::ones((1, 1, 4, 64), candle_core::DType::F16, &device).unwrap();
        manager.append_kv(7, &key, &value).unwrap();

        assert!(manager.get_cached_key(7).is_some());
        assert!(manager.get_cached_value(7).is_some());
        assert!(manager.get_cached_key(999).is_none());
    }
}
