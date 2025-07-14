//! Configuration module for Moshi language model

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main configuration for the Moshi language model
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Feedforward dimension
    pub dim_feedforward: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Audio configuration
    pub audio: AudioConfig,
    /// Transformer configuration
    pub transformer: crate::transformer::Config,
    /// Conditioning configuration
    pub conditioning: ConditioningConfig,
    /// Streaming configuration
    pub streaming: crate::streaming::StreamingConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            d_model: 768,
            num_heads: 12,
            num_layers: 12,
            dim_feedforward: 3072,
            max_seq_len: 4096,
            vocab_size: 32000,
            audio: AudioConfig::default(),
            transformer: crate::transformer::Config::default(),
            conditioning: ConditioningConfig::default(),
            streaming: crate::streaming::StreamingConfig::default(),
        }
    }
}

/// Audio processing configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AudioConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u32,
    /// Frame size for processing
    pub frame_size: usize,
    /// Hop length for windowing
    pub hop_length: usize,
    /// Number of mel frequency bins
    pub n_mels: usize,
    /// FFT window size
    pub n_fft: usize,
    /// Minimum frequency for mel scale
    pub f_min: f32,
    /// Maximum frequency for mel scale
    pub f_max: Option<f32>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            channels: 1,
            frame_size: 1024,
            hop_length: 256,
            n_mels: 80,
            n_fft: 1024,
            f_min: 0.0,
            f_max: None,
        }
    }
}

/// Configuration for conditioning
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConditioningConfig {
    /// Lookup table conditioners
    pub lut_conditioners: HashMap<String, LutConfig>,
    /// Tensor conditioners  
    pub tensor_conditioners: HashMap<String, TensorConfig>,
    /// Global conditioning dimension
    pub global_dim: Option<usize>,
}

impl Default for ConditioningConfig {
    fn default() -> Self {
        Self {
            lut_conditioners: HashMap::new(),
            tensor_conditioners: HashMap::new(),
            global_dim: None,
        }
    }
}

/// Configuration for lookup table conditioner
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LutConfig {
    /// Output dimension
    pub output_dim: usize,
    /// Number of bins
    pub n_bins: usize,
    /// Embedding dimension
    pub dim: usize,
    /// Possible values
    pub possible_values: Vec<String>,
}

impl Default for LutConfig {
    fn default() -> Self {
        Self {
            output_dim: 256,
            n_bins: 100,
            dim: 128,
            possible_values: vec![],
        }
    }
}

/// Configuration for tensor conditioner
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TensorConfig {
    /// Input dimension
    pub dim: usize,
}

impl Default for TensorConfig {
    fn default() -> Self {
        Self { dim: 256 }
    }
}

/// Configuration for model generation
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_length: usize,
    /// Temperature for sampling
    pub temperature: f64,
    /// Top-k sampling parameter
    pub top_k: Option<usize>,
    /// Top-p (nucleus) sampling parameter
    pub top_p: Option<f64>,
    /// Repetition penalty
    pub repetition_penalty: f64,
    /// Whether to use sampling
    pub do_sample: bool,
    /// Number of beams for beam search
    pub num_beams: usize,
    /// Early stopping for beam search
    pub early_stopping: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.9),
            repetition_penalty: 1.0,
            do_sample: true,
            num_beams: 1,
            early_stopping: true,
        }
    }
}

/// Builder for creating configurations
#[derive(Debug, Default)]
pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    /// Create a new config builder
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the model dimension
    pub fn d_model(mut self, d_model: usize) -> Self {
        self.config.d_model = d_model;
        self.config.transformer.d_model = d_model;
        self
    }
    
    /// Set the number of attention heads
    pub fn num_heads(mut self, num_heads: usize) -> Self {
        self.config.num_heads = num_heads;
        self.config.transformer.num_heads = num_heads;
        self
    }
    
    /// Set the number of transformer layers
    pub fn num_layers(mut self, num_layers: usize) -> Self {
        self.config.num_layers = num_layers;
        self.config.transformer.num_layers = num_layers;
        self
    }
    
    /// Set the vocabulary size
    pub fn vocab_size(mut self, vocab_size: usize) -> Self {
        self.config.vocab_size = vocab_size;
        self
    }
    
    /// Set the maximum sequence length
    pub fn max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.config.max_seq_len = max_seq_len;
        self.config.transformer.max_seq_len = max_seq_len;
        self
    }
    
    /// Set the audio configuration
    pub fn audio_config(mut self, audio_config: AudioConfig) -> Self {
        self.config.audio = audio_config;
        self
    }
    
    /// Set the transformer configuration
    pub fn transformer_config(mut self, transformer_config: crate::transformer::Config) -> Self {
        self.config.transformer = transformer_config;
        self
    }
    
    /// Set the conditioning configuration
    pub fn conditioning_config(mut self, conditioning_config: ConditioningConfig) -> Self {
        self.config.conditioning = conditioning_config;
        self
    }
    
    /// Build the final configuration
    pub fn build(self) -> Config {
        self.config
    }
}

impl Config {
    /// Create a new config builder
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::new()
    }
    
    /// Load configuration from JSON file
    pub fn from_json_file(path: &str) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to JSON file
    pub fn to_json_file(&self, path: &str) -> crate::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.d_model == 0 {
            return Err(crate::error::MoshiError::Config("d_model must be > 0".to_string()));
        }
        if self.num_heads == 0 {
            return Err(crate::error::MoshiError::Config("num_heads must be > 0".to_string()));
        }
        if self.d_model % self.num_heads != 0 {
            return Err(crate::error::MoshiError::Config("d_model must be divisible by num_heads".to_string()));
        }
        if self.vocab_size == 0 {
            return Err(crate::error::MoshiError::Config("vocab_size must be > 0".to_string()));
        }
        Ok(())
    }
}
