//! Core language model implementation for Moshi

use crate::{
    conditioner::{Condition, ConditionProvider},
    config::Config,
    sampling_config::SamplingConfig,
    streaming::{CaSrc, StreamingModule, StreamingTransformer},
    tokenizer::KyutaiTokenizer,
};
use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::apply_repeat_penalty;
use std::path::Path;

/// Audio output projection for multi-codebook audio generation
#[derive(Debug)]
pub struct AudioOutputProjection {
    /// Audio codebook projections (one per codebook)
    codebook_projections: Vec<Linear>,
    /// Number of audio codebooks
    num_codebooks: usize,
    /// Audio vocabulary size per codebook
    audio_vocab_size: usize,
}

impl AudioOutputProjection {
    /// Create a new audio output projection
    pub fn new(
        d_model: usize,
        audio_vocab_size: usize,
        num_codebooks: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut codebook_projections = Vec::with_capacity(num_codebooks);

        for i in 0..num_codebooks {
            let proj = candle_nn::linear(
                d_model,
                audio_vocab_size,
                vb.pp(&format!("audio_proj_{}", i)),
            )?;
            codebook_projections.push(proj);
        }

        Ok(Self {
            codebook_projections,
            num_codebooks,
            audio_vocab_size,
        })
    }

    /// Forward pass through all codebook projections
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Vec<Tensor>> {
        let mut audio_logits = Vec::with_capacity(self.num_codebooks);

        for projection in &self.codebook_projections {
            let logits = hidden_states.apply(projection)?;
            audio_logits.push(logits);
        }

        Ok(audio_logits)
    }

    /// Get the number of codebooks
    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// Get the audio vocabulary size
    pub fn audio_vocab_size(&self) -> usize {
        self.audio_vocab_size
    }
}

/// Core Moshi language model
#[derive(Debug)]
pub struct LmModel {
    /// Token embedding layer
    embed_tokens: Embedding,
    /// Main transformer
    transformer: StreamingTransformer,
    /// Output projection layer
    output_proj: Linear,
    /// Audio output projection for multi-codebook generation
    audio_output_proj: AudioOutputProjection,
    /// Condition provider for conditioning
    condition_provider: Option<ConditionProvider>,
    /// Model configuration
    config: Config,
    /// Model device
    device: Device,
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

    /// Generate text using the model with default sampling configuration
    pub fn generate(
        &mut self,
        input_ids: &Tensor,
        max_length: usize,
        temperature: f64,
        top_k: Option<usize>,
        conditions: Option<&[Condition]>,
    ) -> Result<Tensor> {
        // Convert legacy parameters to SamplingConfig
        let sampling = match top_k {
            Some(k) => Sampling::TopK { k, temperature },
            None => Sampling::All { temperature },
        };
        let config = SamplingConfig::custom(sampling, Some(1.1), 64, 42);
        self.generate_with_config(input_ids, max_length, &config, conditions)
    }

    /// Generate text using the model with advanced sampling configuration
    pub fn generate_with_config(
        &mut self,
        input_ids: &Tensor,
        max_length: usize,
        config: &SamplingConfig,
        conditions: Option<&[Condition]>,
    ) -> Result<Tensor> {
        let mut generated = input_ids.clone();
        let mut context: Vec<u32> = Vec::new();

        // Extract initial context from input_ids if possible
        if let Ok(input_vec) = input_ids.to_vec2::<u32>() {
            if !input_vec.is_empty() {
                context = input_vec[0].clone();
            }
        }

        for _ in 0..max_length {
            // Forward pass
            let logits = self.forward(&generated, conditions)?;

            // Get last token logits
            let seq_len = logits.dim(1)?;
            let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

            // Sample next token using advanced sampling
            let next_token_id = self.sample_from_logits(&last_logits, config, &context)?;

            // Update context for repetition penalty
            context.push(next_token_id);

            // Convert token ID to tensor and concatenate
            let next_token = Tensor::new(&[next_token_id], &self.device)?
                .unsqueeze(0)?
                .unsqueeze(1)?;
            generated = Tensor::cat(&[&generated, &next_token], 1)?;
        }

        Ok(generated)
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
    pub fn reset_streaming(&mut self) {
        self.transformer.reset_streaming();
    }

    /// Get model configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get model device
    pub fn device(&self) -> &Device {
        &self.device
    }

    // ASR-compatible methods

    /// Get the text start token ID
    pub fn text_start_token(&self) -> u32 {
        1 // Common text start token
    }

    /// Get the audio pad token ID
    pub fn audio_pad_token(&self) -> u32 {
        0 // Common padding token
    }

    /// Get the number of input audio codebooks
    pub fn in_audio_codebooks(&self) -> usize {
        8 // Standard number for Moshi audio codebooks
    }

    /// Reset the model state for streaming
    pub fn reset_state(&mut self) {
        self.reset_streaming();
    }

    /// ASR-compatible forward method
    pub fn forward_asr(
        &mut self,
        text: Option<Tensor>,
        audio_tokens: Vec<Option<Tensor>>,
    ) -> Result<(Tensor, Tensor)> {
        // Handle input preparation
        let batch_size = 1;
        let seq_len = 1;

        // Create or use provided text input
        let text_input = if let Some(text) = text {
            text
        } else {
            // Use padding token if no text provided
            Tensor::from_vec(
                vec![self.text_start_token()],
                (batch_size, seq_len),
                &self.device,
            )?
        };

        // Embed text tokens
        let mut hidden_states = text_input.apply(&self.embed_tokens)?;

        // Process audio tokens if provided
        if !audio_tokens.is_empty() {
            if let Some(audio_embeddings) = self.process_audio_tokens(&audio_tokens)? {
                hidden_states =
                    self.fuse_text_audio_representations(&hidden_states, &audio_embeddings)?;
            }
        }

        // Forward through transformer
        let output = self.transformer.forward(&hidden_states)?;

        // Project to vocabulary for text logits
        let text_logits = output.apply(&self.output_proj)?;

        // Generate proper audio logits using multi-codebook projection
        let audio_logits_vec = self.audio_output_proj.forward(&output)?;

        // For backward compatibility, return the first codebook as the main audio logits
        // In a full implementation, this method signature should be updated to return Vec<Tensor>
        let audio_logits = if !audio_logits_vec.is_empty() {
            audio_logits_vec[0].clone()
        } else {
            // Fallback to zeros if no codebooks (shouldn't happen)
            Tensor::zeros_like(&text_logits)?
        };

        Ok((text_logits, audio_logits))
    }

    /// ASR-compatible forward method returning all audio codebook logits
    pub fn forward_asr_multi_codebook(
        &mut self,
        text: Option<Tensor>,
        audio_tokens: Vec<Option<Tensor>>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        // Handle input preparation
        let batch_size = 1;
        let seq_len = 1;

        // Create or use provided text input
        let text_input = if let Some(text) = text {
            text
        } else {
            // Use padding token if no text provided
            Tensor::from_vec(
                vec![self.text_start_token()],
                (batch_size, seq_len),
                &self.device,
            )?
        };

        // Embed text tokens
        let mut hidden_states = text_input.apply(&self.embed_tokens)?;

        // Process audio tokens if provided
        if !audio_tokens.is_empty() {
            if let Some(audio_embeddings) = self.process_audio_tokens(&audio_tokens)? {
                hidden_states =
                    self.fuse_text_audio_representations(&hidden_states, &audio_embeddings)?;
            }
        }

        // Forward through transformer
        let output = self.transformer.forward(&hidden_states)?;

        // Project to vocabulary for text logits
        let text_logits = output.apply(&self.output_proj)?;

        // Generate proper audio logits using multi-codebook projection
        let audio_logits_vec = self.audio_output_proj.forward(&output)?;

        Ok((text_logits, audio_logits_vec))
    }

    /// Get audio output projection information
    pub fn audio_projection_info(&self) -> (usize, usize) {
        (
            self.audio_output_proj.num_codebooks(),
            self.audio_output_proj.audio_vocab_size(),
        )
    }

    // Helper methods

    /// Process audio tokens with proper multi-codebook handling
    fn process_audio_tokens(&self, audio_tokens: &[Option<Tensor>]) -> Result<Option<Tensor>> {
        if audio_tokens.is_empty() {
            return Ok(None);
        }

        let mut codebook_embeddings = Vec::new();
        let mut max_seq_len = 0;
        let batch_size = 1; // From current implementation

        // Process each codebook
        for (codebook_idx, maybe_tokens) in audio_tokens.iter().enumerate() {
            if let Some(tokens) = maybe_tokens {
                // Embed tokens for this codebook
                let embedded = tokens.apply(&self.embed_tokens)?; // Reuse existing embedding
                max_seq_len = max_seq_len.max(tokens.dim(1)?);
                codebook_embeddings.push((codebook_idx, embedded));
            }
        }

        if codebook_embeddings.is_empty() {
            return Ok(None);
        }

        // Combine all codebook embeddings (sum like Mimi decoder)
        let d_model = self.config.d_model;
        let mut combined_embedding = Tensor::zeros(
            (batch_size, max_seq_len, d_model),
            codebook_embeddings[0].1.dtype(),
            &self.device,
        )?;

        for (_idx, embedding) in codebook_embeddings {
            // Pad/truncate to max_seq_len if needed
            let seq_len = embedding.dim(1)?;
            let padded_embedding = if seq_len < max_seq_len {
                let padding = Tensor::zeros(
                    (batch_size, max_seq_len - seq_len, d_model),
                    embedding.dtype(),
                    &self.device,
                )?;
                Tensor::cat(&[&embedding, &padding], 1)?
            } else if seq_len > max_seq_len {
                embedding.narrow(1, 0, max_seq_len)?
            } else {
                embedding
            };

            // Sum embeddings (following Mimi decode pattern)
            combined_embedding = (combined_embedding + padded_embedding)?;
        }

        Ok(Some(combined_embedding))
    }

    /// Fuse text and audio representations
    fn fuse_text_audio_representations(
        &self,
        text_hidden: &Tensor,
        audio_hidden: &Tensor,
    ) -> Result<Tensor> {
        // Strategy 1: Addition (current approach)
        text_hidden.broadcast_add(audio_hidden)
    }

    fn top_k_filter(&self, logits: &Tensor, k: usize) -> Result<Tensor> {
        let vocab_size = logits.dim(1)?;
        if k >= vocab_size {
            return Ok(logits.clone());
        }

        // Implement proper top-k sampling using Candle operations
        let batch_size = logits.dim(0)?;
        let mut filtered_logits = Vec::new();

        for batch_idx in 0..batch_size {
            let batch_logits = logits.get(batch_idx)?;

            // Get the top-k values and indices using Candle operations
            // First convert to vec, then sort to find top-k
            let logits_vec = batch_logits.to_vec1::<f32>()?;
            let mut indexed_logits: Vec<(usize, f32)> = logits_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            
            // Sort by logits value in descending order
            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            // Take top-k
            let top_k_pairs: Vec<(usize, f32)> = indexed_logits.into_iter().take(k).collect();
            let top_indices_vec: Vec<u32> = top_k_pairs.iter().map(|(i, _)| *i as u32).collect();
            let top_values_vec: Vec<f32> = top_k_pairs.iter().map(|(_, v)| *v).collect();

            // Create a mask tensor filled with negative infinity
            let neg_inf = f32::NEG_INFINITY;
            let mut mask = vec![neg_inf; vocab_size];

            // Set the top-k positions to their original values

            for (idx, &token_idx) in top_indices_vec.iter().enumerate() {
                mask[token_idx as usize] = top_values_vec[idx];
            }

            // Convert back to tensor
            let filtered_batch = Tensor::from_vec(mask, (vocab_size,), logits.device())?;
            filtered_logits.push(filtered_batch);
        }

        // Stack the filtered logits back into a batch
        Tensor::stack(&filtered_logits, 0)
    }

    /// Sample from logits using advanced sampling strategies with repetition penalty
    fn sample_from_logits(
        &self,
        logits: &Tensor,
        config: &SamplingConfig,
        context: &[u32],
    ) -> Result<u32> {
        // Apply repetition penalty if configured
        let mut processed_logits = if let Some(penalty) = config.repetition_penalty {
            let context_slice = if context.len() > config.repetition_context_size {
                &context[context.len() - config.repetition_context_size..]
            } else {
                context
            };
            apply_repeat_penalty(logits, penalty, context_slice)?
        } else {
            logits.clone()
        };

        // Apply top-k filtering if using TopK sampling
        if let Sampling::TopK { k, .. } = &config.sampling {
            processed_logits = self.top_k_filter(&processed_logits, *k)?;
        }

        // Use LogitsProcessor for advanced sampling
        let mut processor = LogitsProcessor::from_sampling(config.seed, config.sampling.clone());
        processor.sample(&processed_logits)
    }
}

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

/// Utility functions for model operations
pub mod utils {
    use super::*;

    /// Load model weights from safetensors file
    pub fn load_model_weights(model_path: &str, _device: &Device) -> Result<candle_nn::VarMap> {
        let mut var_map = candle_nn::VarMap::new();
        var_map.load(model_path)?;
        Ok(var_map)
    }

    /// Save model weights to safetensors file
    pub fn save_model_weights(var_map: &candle_nn::VarMap, model_path: &str) -> Result<()> {
        var_map.save(model_path)?;
        Ok(())
    }

    /// Create a production tokenizer from file
    pub fn create_tokenizer<P: AsRef<Path>>(tokenizer_path: P) -> crate::Result<KyutaiTokenizer> {
        KyutaiTokenizer::from_file(tokenizer_path)
    }

    /// Create tokenizer from pretrained model
    #[cfg(feature = "http")]
    pub fn create_pretrained_tokenizer(model_name: &str) -> crate::Result<KyutaiTokenizer> {
        KyutaiTokenizer::from_pretrained(model_name)
    }
}
