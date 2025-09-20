//! Configuration system for the Kyutai language model
//!
//! This module provides a comprehensive configuration system with builders and validation
//! for the Kyutai language model implementation. It includes configurations for transformers,
//! audio processing, conditioning, and model-specific parameters.

pub mod audio_config;
pub mod basic_types;
pub mod conditioners;
pub mod lm_2025;
pub mod lm_base;
pub mod lm_v01_core;
pub mod lm_v01_variants;
pub mod main_config;
pub mod tts_configs;

// Re-export main types for public API
pub use audio_config::AudioConfig;
pub use basic_types::{ConditionerConfig, ConditionersConfig, LutConfig, TensorConfig};
pub use conditioners::{DepFormerConfig, FuserConfig};
pub use lm_base::LmConfig;
pub use main_config::{ConditioningConfig, Config};
pub use tts_configs::TtsConfig;
