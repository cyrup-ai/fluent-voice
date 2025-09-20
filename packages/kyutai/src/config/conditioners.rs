//! Conditioner configurations for model control and fusion

use super::basic_types::ConditionersConfig;
use crate::transformer::Config as TransformerConfig;
use serde::{Deserialize, Serialize};

/// Configuration for fusing different conditioning signals
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FuserConfig {
    pub cross_attention_pos_emb: bool,
    pub cross_attention_pos_emb_scale: f32,
    pub sum: Vec<String>,
    pub prepend: Vec<String>,
    pub cross: Vec<String>,
}

/// Configuration for dependency transformer (DepFormer)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DepFormerConfig {
    pub transformer: TransformerConfig,
    pub num_slices: usize,
    pub low_rank_embeddings: Option<usize>,
    pub shared: bool,
    pub multi_linear: bool,
    pub weights_per_step: bool,
    pub pos_emb: String,
    pub weights_per_step_schedule: Option<Vec<usize>>,
}

/// Main conditioner configuration that combines all conditioning components
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// Basic conditioner configurations mapping
    pub conditioners: ConditionersConfig,
    /// Fuser configuration for signal fusion
    pub fuser: Option<FuserConfig>,
    /// DepFormer configuration for dependency modeling
    pub depformer: Option<DepFormerConfig>,
}
