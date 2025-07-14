//! Moshi Language Model Implementation for Fluent Voice
//! 
//! This crate provides a Rust implementation of the Moshi Language Model System
//! 
//! A comprehensive language model system built with the Candle framework,
//! providing state-of-the-art text generation capabilities with conditioning support.

// Core modules
pub mod asr;
pub mod conditioner;
pub mod config;
pub mod error;
pub mod mimi;
pub mod model;
pub mod streaming;
pub mod transformer;

// Re-exports for convenient access
pub use asr::{State as AsrState, StateBuilder as AsrStateBuilder, Word};
pub use conditioner::{
    Condition, ConditionProvider, Conditioner, LutConditioner, TensorConditioner,
};
pub use config::{
    AudioConfig, Config, ConfigBuilder, ConditioningConfig, GenerationConfig, LutConfig,
    TensorConfig,
};
pub use error::{MoshiError, Result};
pub use mimi::{Config as MimiConfig, Mimi, MimiBuilder, OptionalTensor};
pub use model::{LmModel, LmModelBuilder, SimpleTokenizer};
pub use streaming::{
    CaSrc, StreamTensor, StreamingConfig, StreamingModule, StreamingTransformer,
};
pub use transformer::{
    Config as TransformerConfig, CrossAttentionGating, FeedForward, MultiHeadAttention, Norm,
    NormType, PositionalEmbedding, Transformer, TransformerLayer,
};

// External crate re-exports for convenience
pub use candle;
pub use candle_nn;

// Type aliases for common Candle types
pub type Tensor = candle::Tensor;
pub type Device = candle::Device;
pub type DType = candle::DType;
pub type VarBuilder<'a> = candle_nn::VarBuilder<'a>;
pub type VarMap = candle_nn::VarMap;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        asr::{AsrState, AsrStateBuilder, Word},
        config::{Config, ConfigBuilder},
        error::{MoshiError, Result},
        mimi::{Mimi, MimiBuilder, MimiConfig},
        model::{LmModel, LmModelBuilder},
        streaming::{StreamingConfig, StreamingTransformer},
        transformer::Config as TransformerConfig,
        Tensor, Device, DType, VarBuilder,
    };
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default model configurations
pub mod defaults {
    use super::*;
    
    /// Small model configuration (suitable for testing)
    pub fn small_config() -> Config {
        Config::builder()
            .d_model(256)
            .num_heads(4)
            .num_layers(4)
            .vocab_size(8000)
            .max_seq_len(1024)
            .build()
    }
    
    /// Base model configuration
    pub fn base_config() -> Config {
        Config::default()
    }
    
    /// Large model configuration
    pub fn large_config() -> Config {
        Config::builder()
            .d_model(1024)
            .num_heads(16)
            .num_layers(24)
            .vocab_size(50000)
            .max_seq_len(8192)
            .build()
    }
}

// Convenient type aliases
pub type Result<T> = std::result::Result<T, MoshiError>;
