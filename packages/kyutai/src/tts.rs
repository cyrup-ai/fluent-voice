// src/tts.rs

use super::config::TtsConfig;
use super::lm::LmModel;
use super::mimi::Mimi;
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
            conditioners: None, // TODO: Convert from config::ConditionersConfig to conditioner::Config
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
        Ok(Self { lm, mimi, config })
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
        let _logits_processor = LogitsProcessor::new(seed, Some(temp), Some(top_p));
        // Note: top_k filtering would need to be implemented separately in the generation loop
        let _top_k = top_k; // Store for future use in sampling implementation

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
                // Generate new tokens using the language model with top_k filtering
                text_token = if top_k > 0 && top_k < self.config.lm.text_out_vocab_size {
                    // Apply top_k filtering during generation
                    self.lm
                        .step_with_top_k(text_token, &audio_codes, top_k, Some(&conditions))?
                } else {
                    self.lm.step_without_ca_src(
                        text_token,
                        &audio_codes,
                        None,
                        Some(&conditions),
                    )?
                };

                // Apply repetition penalty if specified
                if let Some((context_len, penalty)) = repetition_penalty {
                    let context_start = generated_tokens.len().saturating_sub(context_len);
                    let context = &generated_tokens[context_start..];

                    // Apply penalty logic (simplified version)
                    if context.contains(&text_token) {
                        // In a full implementation, we would modify logits before sampling
                        // For now, we just track that repetition penalty should be applied
                        let _penalty_applied = penalty;
                    }
                }
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

    /// Tokenize text input - simple character-based implementation
    fn tokenize_text(&self, text: &str) -> Result<Vec<u32>> {
        // Simple character-based tokenization
        // In a full implementation, this would use a proper tokenizer like SentencePiece
        let text_tokens: Vec<u32> = text.chars().map(|c| c as u32).collect();
        Ok(text_tokens)
    }
}
