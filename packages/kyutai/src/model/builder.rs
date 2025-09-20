//! Builder pattern for creating language models

use super::core::LmModel;
use crate::config::Config;
use candle_core::Result;
use candle_nn::VarBuilder;

/// Builder for creating language models
#[derive(Debug)]
pub struct LmModelBuilder {
    config: Config,
}

impl LmModelBuilder {
    /// Create a new model builder
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    /// Set the configuration
    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    /// Set the model dimension
    pub fn d_model(mut self, d_model: usize) -> Self {
        self.config.d_model = d_model;
        self
    }

    /// Set the vocabulary size
    pub fn vocab_size(mut self, vocab_size: usize) -> Self {
        self.config.vocab_size = vocab_size;
        self
    }

    /// Set the number of layers
    pub fn num_layers(mut self, num_layers: usize) -> Self {
        self.config.num_layers = num_layers;
        self
    }

    /// Build the model with the given variable builder
    pub fn build(self, vb: VarBuilder) -> Result<LmModel> {
        // Validate configuration
        self.config
            .validate()
            .map_err(|e| candle::Error::msg(format!("Config validation failed: {}", e)))?;

        LmModel::new(&self.config, vb)
    }
}

impl Default for LmModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}
