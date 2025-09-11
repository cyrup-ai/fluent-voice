//! Whisper STT functionality using domain objects for interoperability.

use crate::transcript::Transcript;
use crate::types::TtsChunk;
use fluent_voice_domain::prelude::*;

#[cfg(feature = "microphone")]
use crate::microphone::{self, Model};
#[cfg(not(feature = "microphone"))]
use crate::whisper::Model;
use crate::whisper::{Decoder, Task, WhichModel, token_id};

use anyhow::Result;
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, Config};
use hf_hub::api::sync::Api;
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// Configuration for Whisper model loading and transcription
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_id: String,
    pub which_model: WhichModel,
    pub model_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub seed: u64,
    pub timestamps: bool,
    pub verbose: bool,
    pub temperature: f64,
    pub task: Option<Task>,
    pub language: Option<String>,
    pub quantized: bool,
    pub cpu: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_id: "base".to_string(),
            which_model: WhichModel::Base,
            model_path: None,
            tokenizer_path: None,
            seed: 299792458,
            timestamps: true,
            verbose: false,
            temperature: 0.0,
            task: None,
            language: None,
            quantized: false,
            cpu: false,
        }
    }
}

/// Whisper-based speech-to-text transcriber
pub struct WhisperTranscriber {
    config: ModelConfig,
}

impl WhisperTranscriber {
    pub fn new() -> Result<Self, VoiceError> {
        Self::with_config(ModelConfig::default())
    }

    pub fn with_config(config: ModelConfig) -> Result<Self, VoiceError> {
        Ok(Self { config })
    }

    pub async fn transcribe(&mut self, source: SpeechSource) -> Result<Transcript, VoiceError> {
        match source {
            SpeechSource::File { path, format: _ } => self.transcribe_file(path).await,
            SpeechSource::Microphone {
                backend: _,
                format: _,
                sample_rate: _,
            } => self.transcribe_microphone().await,
            SpeechSource::Memory {
                data,
                format: _,
                sample_rate,
            } => self.transcribe_memory(data, sample_rate).await,
        }
    }

    async fn transcribe_file(&mut self, path: String) -> Result<Transcript, VoiceError> {
        let (model, tokenizer, device, mel_filters) = self.load_model_data().await?;

        let audio_data = self.load_audio_file(&path).await?;
        let mel = self
            .samples_to_mel(&audio_data, &mel_filters, &device)
            .await?;

        let language_token = if self.config.which_model.is_multilingual() {
            match &self.config.language {
                Some(lang) => Some(token_id(&tokenizer, &format!("<|{lang}|>")).map_err(|_| {
                    VoiceError::Configuration(format!("Language {lang} not supported"))
                })?),
                None => Some(token_id(&tokenizer, "<|en|>").map_err(|_| {
                    VoiceError::Configuration("English language token not found".to_string())
                })?),
            }
        } else {
            None
        };

        let mut decoder = Decoder::new(
            model,
            tokenizer,
            self.config.seed,
            &device,
            language_token,
            self.config.task,
            self.config.timestamps,
            self.config.verbose,
        )
        .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

        let segments = decoder
            .run(&mel)
            .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

        let mut transcript = Transcript::new();
        for segment in segments {
            let chunk = TtsChunk::new(
                segment.start,
                segment.start + segment.duration,
                segment.dr.tokens,
                segment.dr.text,
                segment.dr.avg_logprob,
                segment.dr.no_speech_prob,
                segment.dr.temperature,
                segment.dr.compression_ratio,
            );
            transcript.push(chunk);
        }

        Ok(transcript)
    }

    #[cfg(feature = "microphone")]
    async fn transcribe_microphone(&mut self) -> Result<Transcript, VoiceError> {
        microphone::record()
            .await
            .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;
        Ok(Transcript::new())
    }

    #[cfg(not(feature = "microphone"))]
    async fn transcribe_microphone(&mut self) -> Result<Transcript, VoiceError> {
        Err(VoiceError::Configuration(
            "Microphone transcription requires 'microphone' feature".to_string(),
        ))
    }

    async fn transcribe_memory(
        &mut self,
        data: Vec<u8>,
        sample_rate: u32,
    ) -> Result<Transcript, VoiceError> {
        let (model, tokenizer, device, mel_filters) = self.load_model_data().await?;

        let samples = self.bytes_to_samples(data, sample_rate)?;
        let mel = self.samples_to_mel(&samples, &mel_filters, &device).await?;

        let mut decoder = Decoder::new(
            model,
            tokenizer,
            self.config.seed,
            &device,
            None,
            self.config.task,
            self.config.timestamps,
            self.config.verbose,
        )
        .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

        let segments = decoder
            .run(&mel)
            .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

        let mut transcript = Transcript::new();
        for segment in segments {
            let chunk = TtsChunk::new(
                segment.start,
                segment.start + segment.duration,
                segment.dr.tokens,
                segment.dr.text,
                segment.dr.avg_logprob,
                segment.dr.no_speech_prob,
                segment.dr.temperature,
                segment.dr.compression_ratio,
            );
            transcript.push(chunk);
        }

        Ok(transcript)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn is_multilingual(&self) -> bool {
        self.config.which_model.is_multilingual()
    }

    pub fn model_info(&self) -> (&'static str, &'static str) {
        self.config.which_model.model_and_revision()
    }

    async fn load_model_data(&self) -> Result<(Model, Tokenizer, Device, Vec<f32>), VoiceError> {
        let device = if self.config.cpu {
            Device::Cpu
        } else {
            Device::new_metal(0)
                .or_else(|_| Device::new_cuda(0))
                .unwrap_or(Device::Cpu)
        };

        let (default_model, _default_revision) = if self.config.quantized {
            ("lmz/candle-whisper", "main")
        } else {
            self.config.which_model.model_and_revision()
        };

        let model_id = self
            .config
            .model_path
            .as_ref()
            .and_then(|p| p.to_str())
            .unwrap_or(default_model);
        let _revision = "main";

        // Download model using hf-hub
        let api = Api::new().map_err(|e| {
            VoiceError::ProcessingError(format!("Failed to create hf-hub client: {e}"))
        })?;
        let repo = api.model(model_id.to_string());

        let (config_filename, tokenizer_filename, weights_filename) = if self.config.quantized {
            let ext = match self.config.which_model {
                WhichModel::TinyEn => "tiny-en",
                WhichModel::Tiny => "tiny",
                _ => {
                    return Err(VoiceError::Configuration(format!(
                        "No quantized support for {:?}",
                        self.config.which_model
                    )));
                }
            };
            (
                repo.get(&format!("config-{ext}.json")).map_err(|e| {
                    VoiceError::ProcessingError(format!(
                        "Failed to download config-{ext}.json: {e}"
                    ))
                })?,
                repo.get(&format!("tokenizer-{ext}.json")).map_err(|e| {
                    VoiceError::ProcessingError(format!(
                        "Failed to download tokenizer-{ext}.json: {e}"
                    ))
                })?,
                repo.get(&format!("model-{ext}-q80.gguf")).map_err(|e| {
                    VoiceError::ProcessingError(format!(
                        "Failed to download model-{ext}-q80.gguf: {e}"
                    ))
                })?,
            )
        } else {
            (
                repo.get("config.json").map_err(|e| {
                    VoiceError::ProcessingError(format!("Failed to download config.json: {e}"))
                })?,
                repo.get("tokenizer.json").map_err(|e| {
                    VoiceError::ProcessingError(format!("Failed to download tokenizer.json: {e}"))
                })?,
                repo.get("model.safetensors").map_err(|e| {
                    VoiceError::ProcessingError(format!(
                        "Failed to download model.safetensors: {e}"
                    ))
                })?,
            )
        };

        let whisper_config: Config = serde_json::from_str(
            &std::fs::read_to_string(config_filename)
                .map_err(|e| VoiceError::ProcessingError(e.to_string()))?,
        )
        .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

        let model = if self.config.quantized {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &weights_filename,
                &device,
            )
            .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;
            Model::Quantized(
                m::quantized_model::Whisper::load(&vb, whisper_config.clone())
                    .map_err(|e| VoiceError::ProcessingError(e.to_string()))?,
            )
        } else {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)
                    .map_err(|e| VoiceError::ProcessingError(e.to_string()))?
            };
            Model::Normal(
                m::model::Whisper::load(&vb, whisper_config.clone())
                    .map_err(|e| VoiceError::ProcessingError(e.to_string()))?,
            )
        };

        let mel_bytes = match whisper_config.num_mel_bins {
            80 => include_bytes!("melfilters.bytes").as_slice(),
            128 => include_bytes!("melfilters128.bytes").as_slice(),
            nmel => {
                return Err(VoiceError::Configuration(format!(
                    "Unexpected num_mel_bins {nmel}"
                )));
            }
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        LittleEndian::read_f32_into(mel_bytes, &mut mel_filters);

        Ok((model, tokenizer, device, mel_filters))
    }

    async fn load_audio_file(&self, _path: &str) -> Result<Vec<f32>, VoiceError> {
        Ok(vec![0.0; 16000])
    }

    async fn samples_to_mel(
        &self,
        samples: &[f32],
        mel_filters: &[f32],
        device: &Device,
    ) -> Result<Tensor, VoiceError> {
        let config = Config {
            vocab_size: 51864,
            num_mel_bins: 80,
            encoder_layers: 6,
            encoder_attention_heads: 8,
            decoder_layers: 6,
            decoder_attention_heads: 8,
            d_model: 512,
            suppress_tokens: vec![],
            max_target_positions: 448,
            max_source_positions: 1500,
        };

        let mel =
            candle_transformers::models::whisper::audio::pcm_to_mel(&config, samples, mel_filters);

        let mel_len = mel.len();
        let mel_tensor = Tensor::from_vec(
            mel,
            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
            device,
        )
        .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

        Ok(mel_tensor)
    }

    fn bytes_to_samples(&self, data: Vec<u8>, _sample_rate: u32) -> Result<Vec<f32>, VoiceError> {
        let samples = data
            .chunks(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0)]);
                sample as f32 / 32768.0
            })
            .collect();
        Ok(samples)
    }
}
