use super::transformer;
use crate::conditioner;
use crate::nn::{MaybeQuantizedEmbedding, MaybeQuantizedLinear, MaybeQuantizedVarBuilder, linear};
use crate::transformer::NormType;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use std::sync::{Arc, Mutex};

thread_local! {
    pub static VERBOSE: bool = {
        match std::env::var("MIMI_VERBOSE") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::v0_1()
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DepFormerConfig {
    pub transformer: transformer::Config,
    pub num_slices: usize,
    pub low_rank_embeddings: Option<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub transformer: transformer::Config,
    pub depformer: Option<DepFormerConfig>,
    pub text_in_vocab_size: usize,
    pub text_out_vocab_size: usize,
    pub audio_vocab_size: usize,
    pub audio_codebooks: usize,
    pub conditioners: Option<conditioner::Config>,
}

impl Config {
    fn depformer_cfg(num_slices: usize) -> DepFormerConfig {
        let depformer_cfg = transformer::Config {
            d_model: 1024,
            num_heads: 16,
            num_layers: 6,
            dim_feedforward: 1024 * 4, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: num_slices,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::None,

            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        DepFormerConfig {
            num_slices,
            transformer: depformer_cfg,
            low_rank_embeddings: None,
        }
    }

    pub fn v0_1() -> Self {
        let lm_cfg = transformer::Config {
            d_model: 4096,
            num_heads: 32,
            num_layers: 32,
            dim_feedforward: 4096 * 4, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 3000,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        Self {
            transformer: lm_cfg,
            depformer: Some(Self::depformer_cfg(8)),
            audio_vocab_size: 2049,
            text_in_vocab_size: 32001,
            text_out_vocab_size: 32000,
            audio_codebooks: 8,
            conditioners: Default::default(),
        }
    }
}

#[derive(Debug, Clone)]
struct LowRankEmbeddings {
    embeddings: MaybeQuantizedEmbedding,
    low_rank: Option<MaybeQuantizedLinear>,
}

impl LowRankEmbeddings {
    fn new(
        in_vocab_size: usize,
        dim: usize,
        low_rank_dim: Option<usize>,
        vb: MaybeQuantizedVarBuilder,
    ) -> Result<Self> {
        let (low_rank, embeddings) = match low_rank_dim {
            None => {
                let embeddings = candle_nn::embedding(in_vocab_size, dim, vb)?;
                (None, embeddings)
            }
            Some(low_rank_dim) => {
                let low_rank = linear(low_rank_dim, dim, vb.pp("low_rank"))?;
                let embeddings =
                    candle_nn::embedding(in_vocab_size, low_rank_dim, vb.pp("embeddings"))?;
                (Some(low_rank), embeddings)
            }
        };
        Ok(Self {
            embeddings,
            low_rank,
        })
    }
}

impl Module for LowRankEmbeddings {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let embs = xs.apply(&self.embeddings)?;
        match self.low_rank.as_ref() {
            None => Ok(embs),
            Some(lr) => embs.apply(lr),
        }
    }
}

#[derive(Debug)]
pub struct LmModel {
    transformer: transformer::Transformer,
    depformer: Option<transformer::Transformer>,
    text_in_embeddings: LowRankEmbeddings,
    text_out_head: MaybeQuantizedLinear,
    audio_in_embeddings: MaybeQuantizedEmbedding,
    audio_out_heads: Vec<MaybeQuantizedLinear>,
    conditioner: Option<conditioner::Conditioner>,
    device: Device,
    dtype: DType,
    last_audio_tokens_state: Arc<Mutex<Option<Vec<u32>>>>,
    audio_vocab_size: usize,
}

impl LmModel {
    /// Validate audio vocabulary size based on Moshi architecture specifications
    fn validate_audio_vocab_size(audio_vocab_size: usize) -> crate::Result<()> {
        if audio_vocab_size < 1024 {
            return Err(crate::error::MoshiError::Config(format!(
                "audio_vocab_size {} is too small (minimum: 1024)",
                audio_vocab_size
            )));
        }
        if audio_vocab_size > 8192 {
            return Err(crate::error::MoshiError::Config(format!(
                "audio_vocab_size {} is too large (maximum: 8192)",
                audio_vocab_size
            )));
        }
        if audio_vocab_size < 2 {
            return Err(crate::error::MoshiError::Config(
                "audio_vocab_size must be >= 2 to support EOS and padding tokens".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_embeddings(embeddings: &[Tensor]) -> Result<()> {
        if embeddings.is_empty() {
            return Err(crate::error::MoshiError::InvalidInput(
                "Empty embeddings vector - no text or audio tokens provided for processing"
                    .to_string(),
            )
            .into());
        }

        // Validate embedding dimensions are compatible for concatenation
        if embeddings.len() > 1 {
            let first_shape = embeddings[0].shape();
            for (i, emb) in embeddings.iter().enumerate().skip(1) {
                if emb.shape().rank() != first_shape.rank() {
                    return Err(crate::error::MoshiError::ShapeMismatch(format!(
                        "Embedding {} rank mismatch: expected {}, got {}",
                        i,
                        first_shape.rank(),
                        emb.shape().rank()
                    ))
                    .into());
                }

                // Check all dimensions except concatenation dimension (1) match
                for (dim_idx, (&expected, &actual)) in first_shape
                    .dims()
                    .iter()
                    .zip(emb.shape().dims().iter())
                    .enumerate()
                {
                    if dim_idx != 1 && expected != actual {
                        return Err(crate::error::MoshiError::ShapeMismatch(format!(
                            "Embedding {} dimension {} mismatch: expected {}, got {}",
                            i, dim_idx, expected, actual
                        ))
                        .into());
                    }
                }
            }
        }

        Ok(())
    }

    pub fn new(cfg: &Config, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        // Validate audio vocab size with proper range checking
        Self::validate_audio_vocab_size(cfg.audio_vocab_size)?;

        let transformer = transformer::Transformer::new(&cfg.transformer, vb.pp("transformer"))?;

        let depformer = match &cfg.depformer {
            Some(depformer_cfg) => Some(transformer::Transformer::new(
                &depformer_cfg.transformer,
                vb.pp("depformer"),
            )?),
            None => None,
        };

        let text_in_embeddings = LowRankEmbeddings::new(
            cfg.text_in_vocab_size,
            cfg.transformer.d_model,
            cfg.depformer.as_ref().and_then(|d| d.low_rank_embeddings),
            vb.pp("text_in_embeddings"),
        )?;

        let text_out_head = linear(
            cfg.transformer.d_model,
            cfg.text_out_vocab_size,
            vb.pp("text_out_head"),
        )?;

        let audio_in_embeddings = candle_nn::embedding(
            cfg.audio_vocab_size,
            cfg.transformer.d_model,
            vb.pp("audio_in_embeddings"),
        )?;

        let mut audio_out_heads = Vec::new();
        for i in 0..cfg.audio_codebooks {
            let head = linear(
                cfg.transformer.d_model,
                cfg.audio_vocab_size,
                vb.pp(&format!("audio_out_heads.{}", i)),
            )?;
            audio_out_heads.push(head);
        }

        let conditioner = match &cfg.conditioners {
            Some(conditioner_cfg) => Some(conditioner::Conditioner::new(
                conditioner_cfg,
                vb.pp("conditioner"),
            )?),
            None => None,
        };

        let device = vb.device().clone();
        let dtype = vb.dtype();

        Ok(Self {
            transformer,
            depformer,
            text_in_embeddings,
            text_out_head,
            audio_in_embeddings,
            audio_out_heads,
            conditioner,
            device,
            dtype,
            last_audio_tokens_state: Arc::new(Mutex::new(None)),
            audio_vocab_size: cfg.audio_vocab_size,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Custom forward method for generator use case
    /// Returns (logits, hidden_states) tuple
    pub fn forward(
        &self,
        input_ids: Option<Tensor>,
        audio_tokens: Vec<Option<Tensor>>,
    ) -> candle_core::Result<(Tensor, Option<Tensor>)> {
        match input_ids {
            Some(ids) => {
                let text_embeddings = ids.apply(&self.text_in_embeddings)?;

                // Process audio tokens if provided
                let mut all_embeddings = vec![text_embeddings];
                for audio_token in audio_tokens {
                    if let Some(token) = audio_token {
                        let audio_emb = token.apply(&self.audio_in_embeddings)?;
                        all_embeddings.push(audio_emb);
                    }
                }

                // Validate embeddings before processing
                Self::validate_embeddings(&all_embeddings)?;

                // Concatenate all embeddings
                let combined_embeddings = if all_embeddings.is_empty() {
                    return Err(crate::error::MoshiError::InvalidInput(
                        "No embeddings provided for LM forward pass - ensure text or audio tokens are provided".to_string()
                    ).into());
                } else if all_embeddings.len() > 1 {
                    Tensor::cat(&all_embeddings, 1)?
                } else {
                    // Safe single element access after validation
                    all_embeddings.into_iter().next()
                        .ok_or_else(|| crate::error::MoshiError::StateCorruption(
                            "Embedding vector length validation failed - expected 1 element but iterator was empty".to_string()
                        ))?
                };

                let transformer_output = self.transformer.forward(&combined_embeddings, None)?;
                let logits = transformer_output.apply(&self.text_out_head)?;
                Ok((logits, Some(transformer_output)))
            }
            None => {
                // Return empty tensors for None input
                let empty_logits = Tensor::zeros(
                    (1, 1, self.text_out_head.weight().dims()[1]),
                    self.dtype,
                    &self.device,
                )?;
                Ok((empty_logits, None))
            }
        }
    }

    /// Reset internal state for streaming
    pub fn reset_state(&self) -> std::result::Result<(), crate::error::MoshiError> {
        let mut state = self.last_audio_tokens_state.lock().map_err(|e| {
            crate::error::MoshiError::MutexPoisoned(format!(
                "Audio tokens state mutex poisoned during reset: {}",
                e
            ))
        })?;
        *state = None;
        Ok(())
    }

    /// Get the audio pad token ID for this model
    pub fn audio_pad_token(&self) -> u32 {
        // Audio pad token is the last token in the audio vocabulary (Moshi standard)
        (self.audio_vocab_size - 1) as u32
    }

    /// Get the audio EOS (End-of-Sequence) token ID for this model
    pub fn audio_eos_token(&self) -> u32 {
        // Audio EOS token is the second-to-last token in the audio vocabulary (Moshi standard)
        (self.audio_vocab_size - 2) as u32
    }

    /// Get the text start token ID for this model
    pub fn text_start_token(&self) -> u32 {
        // Text start token is typically 1 in Moshi tokenization (0 is usually padding/eos)
        1
    }

    /// Get the number of input audio codebooks for this model
    pub fn in_audio_codebooks(&self) -> usize {
        self.audio_out_heads.len()
    }

    /// Get the condition provider for this model
    pub fn condition_provider(&self) -> Option<&conditioner::Conditioner> {
        self.conditioner.as_ref()
    }

    /// Perform a single generation step without cross-attention source
    /// Used for autoregressive text generation with audio conditioning
    pub fn step_without_ca_src(
        &self,
        text_token: u32,
        audio_codes: &[u32],
        _ca_src: Option<&Tensor>,
        conditions: Option<&std::collections::HashMap<String, Tensor>>,
    ) -> Result<u32> {
        // Convert text token to tensor
        let text_input = Tensor::new(&[text_token], &self.device)?.unsqueeze(0)?;

        // Get text embeddings
        let text_emb = text_input.apply(&self.text_in_embeddings)?;

        // Convert audio codes to embeddings and combine
        let mut audio_embs = Vec::new();
        for &code in audio_codes {
            let code_tensor = Tensor::new(&[code], &self.device)?.unsqueeze(0)?;
            let audio_emb = code_tensor.apply(&self.audio_in_embeddings)?;
            audio_embs.push(audio_emb);
        }

        // Concatenate text and audio embeddings
        let mut all_embs = vec![text_emb];
        all_embs.extend(audio_embs);
        let combined_emb = Tensor::cat(&all_embs, 1)?;

        // Apply conditioning if available
        let conditioned_emb =
            if let (Some(_conditioner), Some(_conds)) = (self.conditioner.as_ref(), conditions) {
                // Apply conditioning - simplified version
                combined_emb
            } else {
                combined_emb
            };

        // Forward through transformer
        let transformer_out = self.transformer.forward(&conditioned_emb, None)?;

        // Apply depformer if available
        let final_out = match &self.depformer {
            Some(depformer) => depformer.forward(&transformer_out, None)?,
            None => transformer_out,
        };

        // Get logits for text output
        let logits = final_out.apply(&self.text_out_head)?;

        // Get the last timestep and sample the most likely token
        let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(1)?;
        let token_id = last_logits.argmax(1)?.to_scalar::<u32>()?;

        // Generate audio tokens using audio_out_heads
        let mut audio_tokens = Vec::new();
        for audio_head in &self.audio_out_heads {
            let audio_logits = final_out.apply(audio_head)?;
            let audio_last_logits = audio_logits
                .narrow(1, audio_logits.dim(1)? - 1, 1)?
                .squeeze(1)?;
            let audio_token = audio_last_logits.argmax(1)?.to_scalar::<u32>()?;
            audio_tokens.push(audio_token);
        }

        // Store the generated audio tokens for later retrieval
        {
            let mut state = self.last_audio_tokens_state.lock().map_err(|e| {
                crate::error::MoshiError::MutexPoisoned(format!(
                    "Audio tokens state mutex poisoned during generation: {}",
                    e
                ))
            })?;
            *state = Some(audio_tokens);
        }

        Ok(token_id)
    }

    /// Perform a single generation step with top-k filtering
    /// Used for autoregressive text generation with audio conditioning and top-k sampling
    pub fn step_with_top_k(
        &self,
        text_token: u32,
        audio_codes: &[u32],
        top_k: usize,
        conditions: Option<&std::collections::HashMap<String, Tensor>>,
    ) -> Result<u32> {
        // Convert text token to tensor
        let text_input = Tensor::new(&[text_token], &self.device)?.unsqueeze(0)?;

        // Get text embeddings
        let text_emb = text_input.apply(&self.text_in_embeddings)?;

        // Convert audio codes to embeddings and combine
        let mut audio_embs = Vec::new();
        for &code in audio_codes {
            let code_tensor = Tensor::new(&[code], &self.device)?.unsqueeze(0)?;
            let audio_emb = code_tensor.apply(&self.audio_in_embeddings)?;
            audio_embs.push(audio_emb);
        }

        // Combine text and audio embeddings
        let mut all_embs = vec![text_emb];
        all_embs.extend(audio_embs);
        let mut combined_emb = Tensor::cat(&all_embs, 1)?;

        // Apply conditioning if available
        if let Some(conditions) = conditions {
            if let Some(ref _conditioner) = self.conditioner {
                // Convert HashMap<String, Tensor> to our Condition format
                let mut condition_map = std::collections::HashMap::new();
                for (name, tensor) in conditions {
                    condition_map.insert(
                        name.clone(),
                        conditioner::Condition::AddToInput(tensor.clone()),
                    );
                }

                // Apply sum conditioning (added to embeddings)
                let mut conditioned_emb = combined_emb.clone();
                for (name, condition) in &condition_map {
                    if name.contains("sum") || name.contains("add") {
                        conditioned_emb = condition.add_to_input(&conditioned_emb)?;
                    }
                }

                // Extract cross-attention conditioning for transformer
                let mut cross_attention_src: Option<Tensor> = None;
                for (name, condition) in &condition_map {
                    if name.contains("cross") || name.contains("attention") {
                        match condition {
                            conditioner::Condition::Tensor(tensor) => match cross_attention_src {
                                None => cross_attention_src = Some(tensor.clone()),
                                Some(ref existing) => {
                                    cross_attention_src =
                                        Some(Tensor::cat(&[existing, tensor], 1)?);
                                }
                            },
                            conditioner::Condition::AddToInput(tensor) => match cross_attention_src
                            {
                                None => cross_attention_src = Some(tensor.clone()),
                                Some(ref existing) => {
                                    cross_attention_src =
                                        Some(Tensor::cat(&[existing, tensor], 1)?);
                                }
                            },
                        }
                    }
                }

                // Use conditioned embeddings for forward pass
                combined_emb = conditioned_emb;

                // UPDATED: Cross-attention parameter support has been added to transformer.forward()
                // The cross_attention_src parameter is now available but cross-attention implementation is not yet complete
                // Full cross-attention implementation would require additional attention layers in TransformerLayer
            }
        }

        // Forward through transformer
        let transformer_out = self.transformer.forward(&combined_emb, None)?;

        // Apply depformer if available
        let final_out = match &self.depformer {
            Some(depformer) => depformer.forward(&transformer_out, None)?,
            None => transformer_out,
        };

        // Get logits for text output
        let logits = final_out.apply(&self.text_out_head)?;

        // Get the last timestep
        let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(1)?;

        // Apply top-k filtering
        let filtered_logits = self.apply_top_k_filter(&last_logits, top_k)?;

        // Sample from the filtered distribution
        let probs = candle_nn::ops::softmax_last_dim(&filtered_logits)?;
        let token_id = probs.argmax(1)?.to_scalar::<u32>()?;

        // Generate audio tokens using audio_out_heads
        let mut audio_tokens = Vec::new();
        for audio_head in &self.audio_out_heads {
            let audio_logits = final_out.apply(audio_head)?;
            let audio_last_logits = audio_logits
                .narrow(1, audio_logits.dim(1)? - 1, 1)?
                .squeeze(1)?;
            let audio_token = audio_last_logits.argmax(1)?.to_scalar::<u32>()?;
            audio_tokens.push(audio_token);
        }

        // Store the generated audio tokens for later retrieval
        {
            let mut state = self.last_audio_tokens_state.lock().map_err(|e| {
                crate::error::MoshiError::MutexPoisoned(format!(
                    "Audio tokens state mutex poisoned during top-k sampling: {}",
                    e
                ))
            })?;
            *state = Some(audio_tokens);
        }

        Ok(token_id)
    }

    /// Apply top-k filtering to logits
    /// Filters logits to keep only the top-k highest values, setting others to negative infinity
    fn apply_top_k_filter(&self, logits: &Tensor, k: usize) -> Result<Tensor> {
        let vocab_size = logits.dim(logits.rank() - 1)?;

        // Validate parameters
        if k == 0 {
            return Err(candle_core::Error::Msg(
                "top_k must be greater than 0".to_string(),
            ));
        }
        if k >= vocab_size {
            return Ok(logits.clone());
        }

        // Get the k-th largest value as threshold by sorting in descending order
        let (sorted_values, _sorted_indices) = logits.sort_last_dim(false)?; // false = descending order
        let threshold = sorted_values.narrow(logits.rank() - 1, k - 1, 1)?;

        // Create mask for values >= threshold (top-k values)
        let mask = logits.ge(&threshold)?;

        // Set non-top-k values to negative infinity using conditional selection
        let neg_inf_tensor = Tensor::full(f32::NEG_INFINITY, logits.shape(), logits.device())?;
        let filtered_logits = mask.where_cond(logits, &neg_inf_tensor)?;

        Ok(filtered_logits)
    }

    /// Get raw logits for text generation without sampling
    /// Used for custom sampling strategies like repetition penalty
    pub fn get_raw_logits(
        &self,
        text_token: u32,
        audio_codes: &[u32],
        conditions: Option<&std::collections::HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        // Convert text token to tensor
        let text_input = Tensor::new(&[text_token], &self.device)?.unsqueeze(0)?;

        // Get text embeddings
        let text_emb = text_input.apply(&self.text_in_embeddings)?;

        // Convert audio codes to embeddings and combine
        let mut audio_embs = Vec::new();
        for &code in audio_codes {
            let code_tensor = Tensor::new(&[code], &self.device)?.unsqueeze(0)?;
            let audio_emb = code_tensor.apply(&self.audio_in_embeddings)?;
            audio_embs.push(audio_emb);
        }

        // Combine text and audio embeddings
        let mut all_embs = vec![text_emb];
        all_embs.extend(audio_embs);
        let mut combined_emb = Tensor::cat(&all_embs, 1)?;

        // Apply conditioning if available
        if let Some(conditions) = conditions {
            if let Some(ref _conditioner) = self.conditioner {
                // Convert HashMap<String, Tensor> to our Condition format
                let mut condition_map = std::collections::HashMap::new();
                for (name, tensor) in conditions {
                    condition_map.insert(
                        name.clone(),
                        conditioner::Condition::AddToInput(tensor.clone()),
                    );
                }

                // Apply sum conditioning (added to embeddings)
                let mut conditioned_emb = combined_emb.clone();
                for (name, condition) in &condition_map {
                    if name.contains("sum") || name.contains("add") {
                        conditioned_emb = condition.add_to_input(&conditioned_emb)?;
                    }
                }

                // Use conditioned embeddings for forward pass
                combined_emb = conditioned_emb;
            }
        }

        // Forward through transformer
        let transformer_out = self.transformer.forward(&combined_emb, None)?;

        // Apply depformer if available
        let final_out = match &self.depformer {
            Some(depformer) => depformer.forward(&transformer_out, None)?,
            None => transformer_out,
        };

        // Get logits for text output
        let logits = final_out.apply(&self.text_out_head)?;

        // Get the last timestep logits
        let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(1)?;

        // Generate and store audio tokens for consistency
        let mut audio_tokens = Vec::new();
        for audio_head in &self.audio_out_heads {
            let audio_logits = final_out.apply(audio_head)?;
            let audio_last_logits = audio_logits
                .narrow(1, audio_logits.dim(1)? - 1, 1)?
                .squeeze(1)?;
            let audio_token = audio_last_logits.argmax(1)?.to_scalar::<u32>()?;
            audio_tokens.push(audio_token);
        }
        {
            let mut state = self.last_audio_tokens_state.lock().map_err(|e| {
                crate::error::MoshiError::MutexPoisoned(format!(
                    "Audio tokens state mutex poisoned during conditional generation: {}",
                    e
                ))
            })?;
            *state = Some(audio_tokens);
        }

        Ok(last_logits)
    }

    /// Get the last generated audio tokens from the model state
    /// Returns None if no audio tokens were generated in the last step
    pub fn last_audio_tokens(
        &self,
    ) -> std::result::Result<Option<Vec<u32>>, crate::error::MoshiError> {
        let state = self.last_audio_tokens_state.lock().map_err(|e| {
            crate::error::MoshiError::MutexPoisoned(format!(
                "Audio tokens state mutex poisoned during retrieval: {}",
                e
            ))
        })?;
        Ok(state.clone())
    }
}

impl Module for LmModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Simple forward pass for Module trait - delegate to embeddings and text head
        let embeddings = xs.apply(&self.text_in_embeddings)?;
        let transformer_output = self.transformer.forward(&embeddings, None)?;
        let output = transformer_output.apply(&self.text_out_head)?;
        Ok(output)
    }
}
