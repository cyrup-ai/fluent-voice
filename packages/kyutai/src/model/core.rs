//! Core language model struct and basic methods

use super::audio_projection::AudioOutputProjection;
use crate::{
    conditioner::{Condition, ConditionProvider},
    config::Config,
    streaming::{CaSrc, StreamingModule, StreamingTransformer},
};
use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

/// Core Moshi language model
#[derive(Debug)]
pub struct LmModel {
    /// Token embedding layer
    pub(super) embed_tokens: Embedding,
    /// Main transformer
    pub(super) transformer: StreamingTransformer,
    /// Output projection layer
    pub(super) output_proj: Linear,
    /// Audio output projection for multi-codebook generation
    pub(super) audio_output_proj: AudioOutputProjection,
    /// Condition provider for conditioning
    pub(super) condition_provider: Option<ConditionProvider>,
    /// Model configuration
    pub(super) config: Config,
    /// Model device
    pub(super) device: Device,
}

impl LmModel {
    /// Create a new language model
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();

        // Create token embeddings
        let embed_tokens =
            candle_nn::embedding(config.vocab_size, config.d_model, vb.pp("embed_tokens"))?;

        // Create transformer
        let transformer = StreamingTransformer::new(&config.transformer, vb.pp("transformer"))?
            .with_streaming_config(config.streaming.clone());

        // Create output projection
        let output_proj =
            candle_nn::linear(config.d_model, config.vocab_size, vb.pp("output_proj"))?;

        // Create audio output projection using configuration values
        let audio_vocab_size = config.lm_config.audio_vocab_size;
        let num_codebooks = config.lm_config.audio_codebooks;
        let audio_output_proj = AudioOutputProjection::new(
            config.d_model,
            audio_vocab_size,
            num_codebooks,
            vb.pp("audio_output_proj"),
        )?;

        // Create condition provider if conditioning is configured
        let condition_provider = if !config.conditioning.lut_conditioners.is_empty()
            || !config.conditioning.tensor_conditioners.is_empty()
        {
            // Convert config to the format expected by ConditionProvider
            let mut conditioner_config = std::collections::HashMap::new();

            for (name, lut_config) in &config.conditioning.lut_conditioners {
                conditioner_config.insert(
                    name.clone(),
                    (
                        "lut".to_string(),
                        lut_config.n_bins,
                        lut_config.dim,
                        lut_config.possible_values.clone(),
                    ),
                );
            }

            for (name, tensor_config) in &config.conditioning.tensor_conditioners {
                conditioner_config.insert(
                    name.clone(),
                    ("tensor".to_string(), 0, tensor_config.dim, vec![]),
                );
            }

            let output_dim = config.conditioning.global_dim.unwrap_or(config.d_model);
            Some(ConditionProvider::new(
                output_dim,
                &conditioner_config,
                vb.pp("condition_provider"),
            )?)
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            transformer,
            output_proj,
            audio_output_proj,
            condition_provider,
            config: config.clone(),
            device,
        })
    }
    /// Forward pass through the model
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        conditions: Option<&[Condition]>,
    ) -> Result<Tensor> {
        // Embed input tokens
        let mut hidden_states = input_ids.apply(&self.embed_tokens)?;

        // Apply conditioning if provided
        if let Some(conditions) = conditions {
            for condition in conditions {
                hidden_states = condition.add_to_input(&hidden_states)?;
            }
        }

        // Forward through transformer
        let hidden_states = self.transformer.forward(&hidden_states)?;

        // Project to vocabulary
        let logits = hidden_states.apply(&self.output_proj)?;

        Ok(logits)
    }

    /// Forward pass with cross-attention source
    pub fn forward_with_ca(
        &mut self,
        input_ids: &Tensor,
        ca_src: Option<&CaSrc>,
        conditions: Option<&[Condition]>,
    ) -> Result<Tensor> {
        // Embed input tokens
        let mut hidden_states = input_ids.apply(&self.embed_tokens)?;

        // Apply conditioning if provided
        if let Some(conditions) = conditions {
            for condition in conditions {
                hidden_states = condition.add_to_input(&hidden_states)?;
            }
        }

        // Forward through transformer with cross-attention
        let hidden_states = self.transformer.forward_ca(&hidden_states, ca_src)?;

        // Project to vocabulary
        let logits = hidden_states.apply(&self.output_proj)?;

        Ok(logits)
    }

    /// Create condition from conditioner
    pub fn create_condition(&self, conditioner_name: &str, value: &str) -> Result<Condition> {
        if let Some(provider) = &self.condition_provider {
            let tensor = provider.get_condition(conditioner_name, value, &self.device)?;
            Ok(Condition::Tensor(tensor))
        } else {
            Err(candle::Error::msg("No condition provider available").into())
        }
    }

    /// Reset streaming state
    pub fn reset_streaming(&mut self) -> std::result::Result<(), crate::error::MoshiError> {
        self.transformer.reset_streaming()?;
        Ok(())
    }

    /// Get model configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get model device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Reset the model state for streaming
    pub fn reset_state(&mut self) -> std::result::Result<(), crate::error::MoshiError> {
        self.reset_streaming()?;
        Ok(())
    }
}
