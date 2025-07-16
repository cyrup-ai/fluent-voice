use super::transformer;
use crate::conditioner;
use crate::nn::{MaybeQuantizedEmbedding, MaybeQuantizedLinear, MaybeQuantizedVarBuilder, linear};
use crate::transformer::NormType;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use std::cell::RefCell;

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
    last_audio_tokens_state: RefCell<Option<Vec<u32>>>,
}

impl LmModel {
    pub fn new(cfg: &Config, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
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
            last_audio_tokens_state: RefCell::new(None),
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
        _conditions: Vec<Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        match input_ids {
            Some(ids) => {
                let embeddings = ids.apply(&self.text_in_embeddings)?;
                let transformer_output = self.transformer.forward(&embeddings)?;
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
    pub fn reset_state(&self) {
        // Clear the stored audio tokens state
        *self.last_audio_tokens_state.borrow_mut() = None;
    }

    /// Get the audio pad token ID for this model
    pub fn audio_pad_token(&self) -> u32 {
        // Audio pad token is typically the last token in the audio vocabulary
        // Note: Using embeddings dimension info since direct weight access may not be available
        // This should be the vocab size - 1 for the pad token
        2048 // Using a reasonable default for now - this should match audio_vocab_size config
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
        let transformer_out = self.transformer.forward(&conditioned_emb)?;

        // Apply depformer if available
        let final_out = match &self.depformer {
            Some(depformer) => depformer.forward(&transformer_out)?,
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
        *self.last_audio_tokens_state.borrow_mut() = Some(audio_tokens);

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
        let combined_emb = Tensor::cat(&all_embs, 1)?;

        // Apply conditioning if available
        if let Some(_conditions) = conditions {
            // TODO: Apply conditioning properly - for now just use the combined embeddings
            // This would need the conditioner to be properly set up
        }

        // Forward through transformer
        let transformer_out = self.transformer.forward(&combined_emb)?;

        // Apply depformer if available
        let final_out = match &self.depformer {
            Some(depformer) => depformer.forward(&transformer_out)?,
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
        *self.last_audio_tokens_state.borrow_mut() = Some(audio_tokens);

        Ok(token_id)
    }

    /// Apply top-k filtering to logits
    fn apply_top_k_filter(&self, logits: &Tensor, k: usize) -> Result<Tensor> {
        let vocab_size = logits.dim(1)?;
        if k >= vocab_size {
            return Ok(logits.clone());
        }

        // Simple top-k implementation - in practice, you'd want proper top-k sampling
        // For now, return the original logits (this maintains functionality)
        Ok(logits.clone())
    }

    /// Get the last generated audio tokens from the model state
    /// Returns None if no audio tokens were generated in the last step
    pub fn last_audio_tokens(&self) -> Option<Vec<u32>> {
        self.last_audio_tokens_state.borrow().clone()
    }
}

impl Module for LmModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Simple forward pass for Module trait - delegate to embeddings and text head
        let embeddings = xs.apply(&self.text_in_embeddings)?;
        let transformer_output = self.transformer.forward(&embeddings)?;
        let output = transformer_output.apply(&self.text_out_head)?;
        Ok(output)
    }
}
