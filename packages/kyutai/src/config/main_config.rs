//! Main configuration struct with validation

use super::audio_config::AudioConfig;
use super::basic_types::{LutConfig, TensorConfig};
use super::lm_base::LmConfig;
use crate::transformer::Config as TransformerConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

/// Main configuration struct (compatible with old interface)
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
    pub transformer: TransformerConfig,
    /// Conditioning configuration
    pub conditioning: ConditioningConfig,
    /// Streaming configuration
    pub streaming: crate::streaming::StreamingConfig,
    /// Language model configuration
    pub lm_config: LmConfig,
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
            transformer: TransformerConfig::default(),
            conditioning: ConditioningConfig::default(),
            streaming: crate::streaming::StreamingConfig::default(),
            lm_config: LmConfig::v0_1(),
        }
    }
}

impl Config {
    /// Validate the configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.d_model == 0 {
            return Err(crate::error::MoshiError::Config(
                "d_model must be > 0".to_string(),
            ));
        }
        if self.num_heads == 0 {
            return Err(crate::error::MoshiError::Config(
                "num_heads must be > 0".to_string(),
            ));
        }
        if self.d_model % self.num_heads != 0 {
            return Err(crate::error::MoshiError::Config(
                "d_model must be divisible by num_heads".to_string(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(crate::error::MoshiError::Config(
                "vocab_size must be > 0".to_string(),
            ));
        }
        if self.lm_config.audio_vocab_size == 0 {
            return Err(crate::error::MoshiError::Config(
                "audio_vocab_size must be > 0".to_string(),
            ));
        }
        if self.lm_config.audio_codebooks == 0 || self.lm_config.audio_codebooks > 64 {
            return Err(crate::error::MoshiError::Config(
                "audio_codebooks must be between 1 and 64".to_string(),
            ));
        }
        Ok(())
    }
}
