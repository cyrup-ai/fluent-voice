// src/tts.rs

use super::config::TtsConfig;
use super::lm::{Config as LmConfig, LmModel};
use super::mimi::Mimi;
use crate::conditioner::Condition;
use crate::error::MoshiError;
use candle::{DType, Device, Result, Tensor};
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

#[derive(Debug, Clone)]
pub struct Model {
    lm: LmModel,
    mimi: Mimi,
    config: Config,
}

impl Model {
    pub fn new(config: Config, lm_vb: VarBuilder, mimi_vb: VarBuilder) -> Result<Self> {
        let lm = LmModel::new(&config.lm, lm_vb)?;
        let mimi = Mimi::new(&config.mimi, mimi_vb)?;
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

        let mut logits_processor = LogitsProcessor::new(seed, Some(temp), Some(top_k), Some(top_p));

        let mut conditions = HashMap::new();
        if let Some(sp) = speaker_pcm {
            conditions.insert("speaker_wavs".to_string(), Condition::Tensor(sp.clone()));
        }
        if let Some(alpha) = cfg_alpha {
            let cfg_value = format!("{:.1}", alpha);
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

        for _ in 0..max_steps {
            text_token =
                self.lm
                    .step_without_ca_src(text_token, &audio_codes, None, Some(&conditions))?;

            if text_token == self.config.tts.text_eop_token {
                break;
            }

            if let Some(codes) = self.lm.last_audio_tokens() {
                let codes_tensor = Tensor::from_vec(
                    codes,
                    (1, 1, self.config.mimi_num_codebooks),
                    self.lm.device(),
                )?;
                let pcm_step = self.mimi.decode_step(&codes_tensor.into())?;
                if let Some(pcm) = pcm_step.as_option() {
                    pcm_out.extend_from_slice(&pcm.to_vec1::<f32>()?);
                }
                audio_codes = codes;
            } else {
                audio_codes = vec![self.lm.audio_pad_token(); self.config.lm.audio_codebooks];
            }
        }

        Ok(pcm_out)
    }
}
