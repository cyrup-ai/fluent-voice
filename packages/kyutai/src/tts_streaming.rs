// src/tts_streaming.rs

use super::tts::Model;
use crate::conditioner::Condition;
use crate::error::MoshiError;
use crate::streaming::StreamingModule;
use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct StreamingModel {
    inner: Arc<Mutex<Model>>,
    state: StreamingState,
    voice_config: Option<VoiceConfig>,
}

#[derive(Debug, Clone)]
pub struct VoiceConfig {
    pub voice_embedding_path: PathBuf,
    pub voice_embedding: Option<Tensor>,
}

#[derive(Debug, Clone)]
struct StreamingState {
    text_token: u32,
    audio_codes: Vec<u32>,
    pcm_buffer: Vec<f32>,
}

impl StreamingModel {
    pub fn new(
        model: Arc<Mutex<crate::tts::Model>>,
    ) -> std::result::Result<Self, crate::error::MoshiError> {
        let (text_start_token, audio_pad_token, audio_codebooks) = {
            let model_ref = model.lock().map_err(|e| {
                crate::error::MoshiError::MutexPoisoned(format!(
                    "TTS model mutex poisoned during initialization: {}",
                    e
                ))
            })?;
            let text_start_token = model_ref.config().tts.text_start_token;
            let audio_pad_token = model_ref.lm().audio_pad_token();
            let audio_codebooks = model_ref.config().lm.audio_codebooks;
            (text_start_token, audio_pad_token, audio_codebooks)
        };

        Ok(Self {
            inner: model,
            state: StreamingState {
                text_token: text_start_token,
                audio_codes: vec![audio_pad_token; audio_codebooks],
                pcm_buffer: vec![],
            },
            voice_config: None,
        })
    }

    /// Configure voice conditioning with voice name from progresshub-downloaded voices
    pub fn with_voice(mut self, voice_name: &str, tts_voices_path: &std::path::Path) -> Self {
        // Build path to specific voice file in progresshub cache
        let voice_path = tts_voices_path.join(format!("{}.safetensors", voice_name));

        self.voice_config = Some(VoiceConfig {
            voice_embedding_path: voice_path,
            voice_embedding: None,
        });
        self
    }

    /// Configure voice conditioning with full path to voice embedding file
    pub fn with_voice_path<P: Into<PathBuf>>(mut self, voice_path: P) -> Self {
        self.voice_config = Some(VoiceConfig {
            voice_embedding_path: voice_path.into(),
            voice_embedding: None,
        });
        self
    }

    /// Load voice embedding from safetensors file
    fn load_voice_embedding(&mut self) -> Result<()> {
        if let Some(ref mut config) = self.voice_config {
            if config.voice_embedding.is_none() {
                let model = self
                    .inner
                    .lock()
                    .map_err(|_| MoshiError::Custom("Lock failed".into()))?;
                let device = model.lm().device().clone();
                drop(model);

                // Load voice embedding using existing safetensors infrastructure
                let voice_vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[&config.voice_embedding_path],
                        DType::F32,
                        &device,
                    )
                    .map_err(|e| {
                        MoshiError::Custom(format!("Failed to load voice embedding: {}", e))
                    })?
                };

                // Load the voice embedding tensor - standard key used in Kyutai voice files
                let voice_embedding = voice_vb
                    .get((1, 512), "embedding")
                    .or_else(|_| voice_vb.get((512,), "embedding"))
                    .or_else(|_| voice_vb.get((1, 1, 512), "voice_embedding"))
                    .map_err(|e| {
                        MoshiError::Custom(format!("Voice embedding not found in file: {}", e))
                    })?;

                config.voice_embedding = Some(voice_embedding);
            }
        }
        Ok(())
    }

    /// Build streaming conditions with voice embedding
    fn build_streaming_conditions(
        &mut self,
        _device: &candle_core::Device,
    ) -> Result<HashMap<String, Condition>> {
        let mut conditions = HashMap::new();

        // Load voice embedding if configured
        if self.voice_config.is_some() {
            self.load_voice_embedding()?;

            if let Some(ref config) = self.voice_config {
                if let Some(ref voice_embedding) = config.voice_embedding {
                    conditions.insert(
                        "voice".to_string(),
                        Condition::Tensor(voice_embedding.clone()),
                    );
                }
            }
        }

        Ok(conditions)
    }

    pub fn step(
        &mut self,
        text: Option<&str>,
        conditions: &HashMap<String, Condition>,
    ) -> Result<Vec<f32>> {
        let mut model = self
            .inner
            .lock()
            .map_err(|_| MoshiError::Custom("Lock failed".into()))?;

        if let Some(text) = text {
            // Tokenize and set new text input
            // Assuming tokenizer is part of model or config
            // For simplicity, assume text is already tokenized to single token for streaming
            self.state.text_token = text
                .parse::<u32>()
                .map_err(|e| MoshiError::Custom(e.to_string()))?;
        }

        // Convert conditions from Condition enum to Tensor
        let tensor_conditions: HashMap<String, Tensor> = conditions
            .iter()
            .map(|(k, v)| {
                let tensor = match v {
                    crate::conditioner::Condition::Tensor(t) => t.clone(),
                    crate::conditioner::Condition::AddToInput(t) => t.clone(),
                };
                (k.clone(), tensor)
            })
            .collect();

        model.lm_mut().step_without_ca_src(
            self.state.text_token,
            &self.state.audio_codes,
            None,
            Some(&tensor_conditions),
        )?;

        if let Some(codes) = model.lm().last_audio_tokens().map_err(|e| {
            crate::error::MoshiError::Custom(format!("Failed to get last audio tokens: {}", e))
        })? {
            let codes_tensor = Tensor::from_vec(
                codes.clone(),
                (1, 1, model.config().mimi_num_codebooks),
                model.lm().device(),
            )?;
            let pcm_step = model.mimi_mut().decode_step(&codes_tensor)?;
            if let Some(pcm) = pcm_step {
                let pcm_vec = pcm.to_vec1::<f32>()?;
                self.state.pcm_buffer.extend_from_slice(&pcm_vec);
                self.state.audio_codes = codes;
            }
        }

        let output = self.state.pcm_buffer.clone();
        self.state.pcm_buffer.clear();

        Ok(output)
    }

    pub fn flush(&mut self) -> Result<Vec<f32>> {
        let mut model = self
            .inner
            .lock()
            .map_err(|_| MoshiError::Custom("Lock failed".into()))?;
        model.mimi_mut().flush()?;
        let output = self.state.pcm_buffer.clone();
        self.state.pcm_buffer.clear();
        Ok(output)
    }

    fn step_internal(
        &mut self,
        conditions: &HashMap<String, crate::conditioner::Condition>,
    ) -> Result<Vec<f32>> {
        let mut model = self
            .inner
            .lock()
            .map_err(|_| MoshiError::Custom("Lock failed".into()))?;

        // Convert conditions from Condition enum to Tensor
        let tensor_conditions: HashMap<String, Tensor> = conditions
            .iter()
            .map(|(k, v)| {
                let tensor = match v {
                    crate::conditioner::Condition::Tensor(t) => t.clone(),
                    crate::conditioner::Condition::AddToInput(t) => t.clone(),
                };
                (k.clone(), tensor)
            })
            .collect();

        model.lm_mut().step_without_ca_src(
            self.state.text_token,
            &self.state.audio_codes,
            None,
            Some(&tensor_conditions),
        )?;

        if let Some(codes) = model.lm().last_audio_tokens().map_err(|e| {
            crate::error::MoshiError::Custom(format!("Failed to get last audio tokens: {}", e))
        })? {
            let codes_tensor = Tensor::from_vec(
                codes.clone(),
                (1, 1, model.config().mimi_num_codebooks),
                model.lm().device(),
            )?;
            let pcm_step = model.mimi_mut().decode_step(&codes_tensor)?;
            if let Some(pcm) = pcm_step {
                let pcm_vec = pcm.to_vec1::<f32>()?;
                self.state.audio_codes = codes;
                return Ok(pcm_vec);
            }
        }

        Ok(vec![])
    }
}

impl StreamingModule for StreamingModel {
    fn forward_streaming(&mut self, input: &Tensor) -> Result<Tensor> {
        // For TTS streaming, input is text tokens, output is audio chunk
        let text_tokens = input.to_vec1::<u32>()?;

        // Build actual voice conditions using loaded embeddings
        let conditions = self.build_streaming_conditions(input.device())?;

        // Process single token from input
        if let Some(&token) = text_tokens.first() {
            self.state.text_token = token;
        }

        let pcm = self.step_internal(&conditions)?;
        let pcm_len = pcm.len();
        Ok(Tensor::from_vec(pcm, (1, pcm_len), input.device())?)
    }

    fn reset_streaming(&mut self) -> std::result::Result<(), crate::error::MoshiError> {
        let (text_start_token, audio_pad_token, audio_codebooks) = {
            let model = self.inner.lock().map_err(|e| {
                crate::error::MoshiError::MutexPoisoned(format!(
                    "TTS model mutex poisoned during streaming reset: {}",
                    e
                ))
            })?;
            (
                model.config().tts.text_start_token,
                model.lm().audio_pad_token(),
                model.config().lm.audio_codebooks,
            )
        };

        self.state.text_token = text_start_token;
        self.state.audio_codes = vec![audio_pad_token; audio_codebooks];
        self.state.pcm_buffer.clear();
        Ok(())
    }

    fn streaming_state_size(&self) -> usize {
        self.state.pcm_buffer.len() + self.state.audio_codes.len()
    }
}
