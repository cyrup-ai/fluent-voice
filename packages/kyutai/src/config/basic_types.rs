//! Basic configuration types for the Kyutai language model

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Lookup table configuration for conditioners
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LutConfig {
    pub n_bins: usize,
    pub dim: usize,
    pub tokenizer: String,
    pub possible_values: Vec<String>,
}

/// Tensor configuration for conditioners
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TensorConfig {
    pub dim: usize,
}

/// Conditioner configuration enum supporting different types
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ConditionerConfig {
    Lut(LutConfig),
    Tensor(TensorConfig),
}

/// Type alias for conditioners configuration mapping
pub type ConditionersConfig = HashMap<String, ConditionerConfig>;
