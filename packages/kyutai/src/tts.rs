// src/tts.rs

use super::config::TtsConfig;
use super::lm::LmModel;
use super::mimi::Mimi;
use super::tokenizer::KyutaiTokenizer;
use crate::conditioner::Condition;
use crate::config::LmConfig;
use crate::error::MoshiError;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Config {
    pub lm: LmConfig,
    pub tts: TtsConfig,
    pub mimi_num_codebooks: usize,
}

impl Config {
    pub fn v202501() -> Self {
        Self {
            lm: LmConfig::tts_1_6b_en_fr(),
            tts: TtsConfig::v202501(),
            mimi_num_codebooks: 32,
        }
    }
}

#[derive(Debug)]
pub struct Model {
    lm: LmModel,
    mimi: Mimi,
    config: Config,
    tokenizer: KyutaiTokenizer,
}

impl Model {
    pub fn new(config: Config, lm_vb: VarBuilder, mimi_vb: VarBuilder) -> Result<Self> {
        // Convert LmConfig to lm::Config
        let lm_depformer =
            config
                .lm
                .depformer
                .as_ref()
                .map(|depformer_cfg| super::lm::DepFormerConfig {
                    transformer: depformer_cfg.transformer.clone(),
                    num_slices: depformer_cfg.num_slices,
                    low_rank_embeddings: depformer_cfg.low_rank_embeddings,
                });

        let lm_config = super::lm::Config {
            transformer: config.lm.transformer.clone(),
            depformer: lm_depformer,
            text_in_vocab_size: config.lm.text_in_vocab_size,
            text_out_vocab_size: config.lm.text_out_vocab_size,
            audio_vocab_size: config.lm.audio_vocab_size,
            audio_codebooks: config.lm.audio_codebooks,
            conditioners: config.lm.conditioners.as_ref().map(|conditioners_config| {
                let mut conditions = std::collections::HashMap::new();
                for (name, conditioner_config) in conditioners_config {
                    match conditioner_config {
                        super::config::ConditionerConfig::Lut(lut_config) => {
                            conditions.insert(
                                name.clone(),
                                (
                                    lut_config.tokenizer.clone(),
                                    lut_config.n_bins,
                                    lut_config.dim,
                                    lut_config.possible_values.clone(),
                                ),
                            );
                        }
                        super::config::ConditionerConfig::Tensor(tensor_config) => {
                            conditions.insert(
                                name.clone(),
                                (
                                    "tensor".to_string(),
                                    0, // n_bins not applicable for tensor conditioner
                                    tensor_config.dim,
                                    vec![], // possible_values not applicable for tensor conditioner
                                ),
                            );
                        }
                    }
                }
                super::conditioner::Config { conditions }
            })
        };
        let lm = LmModel::new(&lm_config, lm_vb)?;
        // Create default mimi config based on mimi_num_codebooks
        let mimi_config = super::mimi::Config {
            channels: 1,
            sample_rate: 24000.0,
            frame_rate: 75.0,
            renormalize: true,
            resample_method: super::mimi::ResampleMethod::Conv,
            transformer: super::transformer::Config::default(),
            quantizer_n_q: config.mimi_num_codebooks,
            quantizer_bins: 2048,
            quantizer_dim: 1024,
        };
        let mimi = Mimi::new(mimi_config, mimi_vb)?;

        // Create tokenizer - try pretrained first if http feature is available
        let tokenizer = {
            #[cfg(feature = "http")]
            {
                KyutaiTokenizer::from_pretrained("kyutai/moshika-pytorch-bf16")
                    .or_else(|_| {
                        tracing::warn!("Failed to load Kyutai tokenizer, trying GPT-2 fallback");
                        KyutaiTokenizer::from_pretrained("gpt2")
                    })
                    .map_err(|e| candle_core::Error::Msg(format!("Failed to initialize tokenizer: {}", e)))?
            }
            #[cfg(not(feature = "http"))]
            {
                // Without http feature, return an error - tokenizer file must be provided
                return Err(candle_core::Error::Msg(
                    "Tokenizer initialization requires either 'http' feature for pretrained models or use load_with_tokenizer() with a tokenizer file".to_string()
                ));
            }
        };

        Ok(Self {
            lm,
            mimi,
            config,
            tokenizer,
        })
    }

    pub fn load<P: AsRef<Path>>(
        lm_model_file: P,
        mimi_model_file: P,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let config = Config::v202501();
        let lm_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[lm_model_file], dtype, dev)? };
        let mimi_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[mimi_model_file], dtype, dev)? };
        Self::new(config, lm_vb, mimi_vb)
    }

    /// Load model with custom tokenizer from file
    pub fn load_with_tokenizer<P: AsRef<Path>>(
        lm_model_file: P,
        mimi_model_file: P,
        tokenizer_file: P,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let config = Config::v202501();
        let lm_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[lm_model_file], dtype, dev)? };
        let mimi_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[mimi_model_file], dtype, dev)? };

        // Load custom tokenizer from file
        let tokenizer = KyutaiTokenizer::from_file(tokenizer_file).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to load tokenizer from file: {}", e))
        })?;

        // Create model components
        let lm_depformer =
            config
                .lm
                .depformer
                .as_ref()
                .map(|depformer_cfg| super::lm::DepFormerConfig {
                    transformer: depformer_cfg.transformer.clone(),
                    num_slices: depformer_cfg.num_slices,
                    low_rank_embeddings: depformer_cfg.low_rank_embeddings,
                });

        let lm_config = super::lm::Config {
            transformer: config.lm.transformer.clone(),
            depformer: lm_depformer,
            text_in_vocab_size: config.lm.text_in_vocab_size,
            text_out_vocab_size: config.lm.text_out_vocab_size,
            audio_vocab_size: config.lm.audio_vocab_size,
            audio_codebooks: config.lm.audio_codebooks,
            conditioners: None,
        };
        let lm = LmModel::new(&lm_config, lm_vb)?;

        let mimi_config = super::mimi::Config {
            channels: 1,
            sample_rate: 24000.0,
            frame_rate: 75.0,
            renormalize: true,
            resample_method: super::mimi::ResampleMethod::Conv,
            transformer: super::transformer::Config::default(),
            quantizer_n_q: config.mimi_num_codebooks,
            quantizer_bins: 2048,
            quantizer_dim: 1024,
        };
        let mimi = Mimi::new(mimi_config, mimi_vb)?;

        Ok(Self {
            lm,
            mimi,
            config,
            tokenizer,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn lm(&self) -> &LmModel {
        &self.lm
    }

    pub fn lm_mut(&mut self) -> &mut LmModel {
        &mut self.lm
    }

    pub fn mimi(&self) -> &Mimi {
        &self.mimi
    }

    pub fn mimi_mut(&mut self) -> &mut Mimi {
        &mut self.mimi
    }

    pub fn tokenizer(&self) -> &KyutaiTokenizer {
        &self.tokenizer
    }

    pub fn generate(
        &mut self,
        text: &str,
        speaker_pcm: Option<&Tensor>,
        max_steps: usize,
        temp: f64,
        top_k: usize,
        top_p: f64,
        repetition_penalty: Option<(usize, f32)>,
        cfg_alpha: Option<f64>,
        seed: u64,
    ) -> Result<Vec<f32>> {
        self.lm.reset_state();
        self.mimi.reset_state();

        // Create logits processor with proper configuration (top_p for nucleus sampling)
        let mut logits_processor = LogitsProcessor::new(seed, Some(temp), Some(top_p));

        // Implement text tokenization using the tokenize_text method
        let text_tokens = self.tokenize_text(text)?;

        let mut conditions = HashMap::new();
        if let Some(sp) = speaker_pcm {
            conditions.insert("speaker_wavs".to_string(), Condition::Tensor(sp.clone()));
        }
        if let Some(alpha) = cfg_alpha {
            let _cfg_value = format!("{:.1}", alpha); // Used for logging/debugging
            conditions.insert(
                "cfg".to_string(),
                Condition::AddToInput(Tensor::new(alpha as f32, self.lm.device())?),
            );
        } else {
            conditions.insert(
                "cfg".to_string(),
                Condition::AddToInput(Tensor::new(1.0f32, self.lm.device())?),
            );
        }
        let conditions = self
            .lm
            .condition_provider()
            .ok_or(MoshiError::Custom("No condition provider".into()))?
            .condition(&conditions)?;

        let mut text_token = self.config.tts.text_start_token;
        let mut audio_codes = vec![self.lm.audio_pad_token(); self.config.lm.audio_codebooks];
        let mut pcm_out = vec![];
        let mut generated_tokens = vec![text_token];
        let mut text_idx = 0;

        for step in 0..max_steps {
            // Log progress periodically for debugging
            if step % 100 == 0 && step > 0 {
                tracing::debug!("Generation step {}/{}", step, max_steps);
            }
            // Use text tokens as input when available
            if text_idx < text_tokens.len() {
                text_token = text_tokens[text_idx] % self.config.lm.text_out_vocab_size as u32;
                text_idx += 1;
            } else {
                // Generate new tokens with custom sampling including repetition penalty
                text_token = if repetition_penalty.is_some() || (top_k > 0 && top_k < self.config.lm.text_out_vocab_size) {
                    // Get raw logits for custom sampling
                    let mut raw_logits = self.lm.get_raw_logits(text_token, &audio_codes, Some(&conditions))?;
                    
                    // Apply repetition penalty if specified
                    if let Some((context_len, penalty)) = repetition_penalty {
                        let context_start = generated_tokens.len().saturating_sub(context_len);
                        let context = &generated_tokens[context_start..];
                        
                        // Apply repetition penalty by reducing logits for repeated tokens
                        let logits_vec = raw_logits.to_vec1::<f32>()?;
                        let mut modified_logits = logits_vec;
                        
                        for &repeated_token in context {
                            if (repeated_token as usize) < modified_logits.len() {
                                // Reduce logit value for repeated tokens (penalty > 1.0 reduces probability)
                                if penalty != 1.0 {
                                    modified_logits[repeated_token as usize] /= penalty;
                                }
                            }
                        }
                        
                        // Recreate tensor with modified logits
                        raw_logits = Tensor::from_vec(modified_logits, raw_logits.shape(), raw_logits.device())?;
                    }
                    
                    // Apply top-k filtering if specified
                    let filtered_logits = if top_k > 0 && top_k < self.config.lm.text_out_vocab_size {
                        self.apply_top_k_filter(&raw_logits, top_k)?
                    } else {
                        raw_logits
                    };
                    
                    // Sample from the processed logits
                    let probs = candle_nn::ops::softmax_last_dim(&filtered_logits)?;
                    let sampled_token = if temp > 0.0 {
                        // Use temperature sampling via logits processor
                        logits_processor.sample(&probs)?
                    } else {
                        // Use argmax for deterministic sampling
                        probs.argmax(0)?.to_scalar::<u32>()?
                    };
                    sampled_token
                } else {
                    // Use existing optimized methods when no custom sampling needed
                    self.lm.step_without_ca_src(
                        text_token,
                        &audio_codes,
                        None,
                        Some(&conditions),
                    )?
                };
            }

            generated_tokens.push(text_token);

            if text_token == self.config.tts.text_eop_token {
                break;
            }

            if let Some(codes) = self.lm.last_audio_tokens() {
                let codes_tensor = Tensor::from_vec(
                    codes.clone(),
                    (1, 1, self.config.mimi_num_codebooks),
                    self.lm.device(),
                )?;
                let pcm_step = self.mimi.decode_step(&codes_tensor)?;
                if let Some(pcm) = pcm_step {
                    pcm_out.extend_from_slice(&pcm.to_vec1::<f32>()?);
                }
                audio_codes = codes;
            } else {
                audio_codes = vec![self.lm.audio_pad_token(); self.config.lm.audio_codebooks];
            }
        }

        Ok(pcm_out)
    }

    /// Tokenize text input using production KyutaiTokenizer
    fn tokenize_text(&self, text: &str) -> Result<Vec<u32>> {
        // Use the production tokenizer with proper error handling
        let tokens = self
            .tokenizer
            .encode(text, true) // add_special_tokens = true for proper sequence handling
            .map_err(|e| candle_core::Error::Msg(format!("Tokenization failed: {}", e)))?;

        // Validate token IDs against vocabulary size constraints
        let max_vocab_size = self.config.lm.text_out_vocab_size;
        for &token_id in &tokens {
            if token_id >= max_vocab_size as u32 {
                return Err(candle_core::Error::Msg(format!(
                    "Token ID {} exceeds vocabulary size {}. Text: '{}'",
                    token_id, max_vocab_size, text
                )));
            }
        }

        // Log tokenization for debugging (can be removed in production)
        tracing::debug!(
            "Tokenized text '{}' into {} tokens: {:?}",
            text,
            tokens.len(),
            if tokens.len() <= 10 {
                format!("{:?}", tokens)
            } else {
                format!("{:?}...", &tokens[..10])
            }
        );

        Ok(tokens)
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
}
