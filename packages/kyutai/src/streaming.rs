//! Streaming module for Moshi language model
//!
//! Provides streaming functionality for real-time audio processing and generation.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::VecDeque;

/// Trait for streaming modules that can process input incrementally
pub trait StreamingModule {
    /// Process a single step of streaming input
    fn forward_streaming(&mut self, input: &Tensor) -> Result<Tensor>;

    /// Reset the streaming state
    fn reset_streaming(&mut self);

    /// Get the current streaming state size
    fn streaming_state_size(&self) -> usize;
}

/// Configuration for streaming transformer
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StreamingConfig {
    pub chunk_size: usize,
    pub overlap: usize,
    pub max_cache_size: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 256,
            overlap: 32,
            max_cache_size: 1024,
        }
    }
}

/// Streaming transformer implementation
#[derive(Debug)]
pub struct StreamingTransformer {
    pub transformer: crate::transformer::Transformer,
    cache: VecDeque<Tensor>,
    config: StreamingConfig,
    _device: Device,
    _dtype: DType,
}

impl StreamingTransformer {
    pub fn new(config: &crate::transformer::Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let transformer = crate::transformer::Transformer::new(config, vb.pp("transformer"))?;
        let streaming_config = StreamingConfig::default();

        Ok(Self {
            transformer,
            cache: VecDeque::new(),
            config: streaming_config,
            _device: device,
            _dtype: dtype,
        })
    }

    pub fn with_streaming_config(mut self, config: StreamingConfig) -> Self {
        self.config = config;
        self
    }

    /// Forward pass with optional cross-attention source
    pub fn forward_ca(&mut self, input: &Tensor, _ca_src: Option<&CaSrc>) -> Result<Tensor> {
        // For now, ignore cross-attention and just forward through transformer
        // This can be extended to handle cross-attention properly
        self.forward(input)
    }

    /// Standard forward pass
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Forward through transformer
        let output = self.transformer.forward(input, None)?;

        // Update cache for streaming
        self.update_cache(&output)?;

        Ok(output)
    }

    fn update_cache(&mut self, output: &Tensor) -> Result<()> {
        // Add new output to cache
        self.cache.push_back(output.clone());

        // Maintain cache size limit
        while self.cache.len() > self.config.max_cache_size {
            self.cache.pop_front();
        }

        Ok(())
    }

    pub fn reset_cache(&mut self) {
        self.cache.clear();
    }

    pub fn get_cached_output(&self, steps_back: usize) -> Option<&Tensor> {
        if steps_back < self.cache.len() {
            self.cache.get(self.cache.len() - 1 - steps_back)
        } else {
            None
        }
    }
}

impl StreamingModule for StreamingTransformer {
    fn forward_streaming(&mut self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    fn reset_streaming(&mut self) {
        self.reset_cache();
    }

    fn streaming_state_size(&self) -> usize {
        self.cache.len()
    }
}

/// Cross-attention source for conditioning
#[derive(Debug, Clone)]
pub enum CaSrc {
    Tokens(Tensor),
    Embeddings(Tensor),
}

impl CaSrc {
    pub fn tokens(&self) -> Option<&Tensor> {
        match self {
            CaSrc::Tokens(t) => Some(t),
            _ => None,
        }
    }

    pub fn embeddings(&self) -> Option<&Tensor> {
        match self {
            CaSrc::Embeddings(e) => Some(e),
            _ => None,
        }
    }
}

/// Stream tensor for incremental processing
#[derive(Debug)]
pub struct StreamTensor {
    data: VecDeque<Tensor>,
    _chunk_size: usize,
    current_pos: usize,
}

impl StreamTensor {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            data: VecDeque::new(),
            _chunk_size: chunk_size,
            current_pos: 0,
        }
    }

    pub fn add_chunk(&mut self, chunk: Tensor) {
        self.data.push_back(chunk);
    }

    pub fn next_chunk(&mut self) -> Option<Tensor> {
        if !self.data.is_empty() {
            self.current_pos += 1;
            self.data.pop_front()
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.data.clear();
        self.current_pos = 0;
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Create a StreamTensor from a single tensor
    pub fn from_tensor(tensor: Tensor) -> Self {
        let mut stream = Self::new(1); // Single chunk size
        stream.add_chunk(tensor);
        stream
    }

    /// Create an empty StreamTensor
    pub fn empty() -> Self {
        Self::new(1)
    }

    /// Get the current tensor as an option (peek without removing)
    pub fn as_option(&self) -> Option<&Tensor> {
        self.data.front()
    }
}

/// Utility function to add sinusoidal embeddings for positional encoding
pub fn add_sin_embeddings(tensor: &Tensor) -> Result<Tensor> {
    let (_batch_size, seq_len, d_model) = tensor.dims3()?;
    let device = tensor.device();
    let dtype = tensor.dtype();

    let mut pos_encoding = vec![0.0f32; seq_len * d_model];

    for pos in 0..seq_len {
        for i in 0..(d_model / 2) {
            let angle = pos as f32 / 10000_f32.powf(2.0 * i as f32 / d_model as f32);
            pos_encoding[pos * d_model + 2 * i] = angle.sin();
            pos_encoding[pos * d_model + 2 * i + 1] = angle.cos();
        }
    }

    let pos_tensor = Tensor::from_slice(&pos_encoding, (1, seq_len, d_model), device)?
        .to_dtype(dtype)?
        .broadcast_as(tensor.shape())?;

    tensor.broadcast_add(&pos_tensor)
}
