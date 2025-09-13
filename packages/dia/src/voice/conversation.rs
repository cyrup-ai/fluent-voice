//! Conversation for managing voice synthesis with multiple speakers

use anyhow::Result;
use std::sync::Arc;

use super::{PitchNote, Speaker, VocalSpeedMod, VoiceError, VoicePlayer, VoicePool};
use crate::audio::channel_delay;

// GPU optimizations have been simplified - channel_delay_gpu removed

/// A conversation that can generate speech with configured speakers and settings
pub struct Conversation<S: Speaker> {
    text: String,
    speaker: S,
    pool: Arc<VoicePool>,
    speed_modifier: Option<VocalSpeedMod>,
    pitch_range: Option<(PitchNote, PitchNote)>,
}

impl<S: Speaker> Conversation<S> {
    /// Create a new conversation with the given text and speaker (async-aware)
    pub async fn new(text: String, speaker: S, pool: Arc<VoicePool>) -> Result<Self, VoiceError> {
        Ok(Self {
            text,
            speaker,
            pool,
            speed_modifier: None,
            pitch_range: None,
        })
    }

    /// Create a new conversation synchronously for streaming interface
    pub fn new_sync(text: String, speaker: S, pool: Arc<VoicePool>) -> Result<Self, VoiceError> {
        Ok(Self {
            text,
            speaker,
            pool,
            speed_modifier: None,
            pitch_range: None,
        })
    }

    /// Set the speaker for this conversation
    pub fn with_speaker<T: Speaker>(self, speaker: T) -> Conversation<T> {
        Conversation {
            text: self.text,
            speaker,
            pool: self.pool,
            speed_modifier: self.speed_modifier,
            pitch_range: self.pitch_range,
        }
    }

    /// Set the speed modifier
    pub fn with_speed_modifier(mut self, modifier: VocalSpeedMod) -> Self {
        self.speed_modifier = Some(modifier);
        self
    }

    /// Set the pitch range
    pub fn with_pitch_range(mut self, low: PitchNote, high: PitchNote) -> Self {
        self.pitch_range = Some((low, high));
        self
    }

    /// Internal method to generate speech (used by dia_voice.rs)
    pub async fn internal_generate(&self) -> Result<VoicePlayer, VoiceError> {
        use crate::{
            audio::{SAMPLE_RATE, normalize_loudness},
            config::DiaConfig,
            model::DiaModel,
            setup,
        };
        use candle_core::{DType, Device, IndexOp};
        use candle_nn::VarBuilder;
        use candle_transformers::generation::LogitsProcessor;
        use tokenizers::Tokenizer;

        // Get device (prefer GPU if available)
        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else if candle_core::utils::metal_is_available() {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        // Setup models if not already done using existing async context
        let model_paths = setup::setup()
            .await
            .map_err(|e| VoiceError::ConfigError(format!("Model setup failed: {e}")))?;

        // Load Dia model
        let cfg = DiaConfig::default();
        let dtype = if device.is_cuda() || device.is_metal() {
            DType::BF16
        } else {
            DType::F32
        };
        let vb = if model_paths.weights.extension().and_then(|s| s.to_str()) == Some("pth") {
            // Load PyTorch model
            VarBuilder::from_pth(&model_paths.weights, dtype, &device)
                .map_err(|e| VoiceError::ConfigError(format!("Failed to load weights: {e}")))?
        } else {
            // Load SafeTensors model (default)
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[&model_paths.weights], dtype, &device)
                    .map_err(|e| VoiceError::ConfigError(format!("Failed to load weights: {e}")))?
            }
        };
        let dia = DiaModel::new(cfg.clone(), vb, dtype)
            .map_err(|e| VoiceError::GenerationError(format!("Model creation failed: {e}")))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&model_paths.tokenizer)
            .map_err(|e| VoiceError::ConfigError(format!("Failed to load tokenizer: {e}")))?;

        // Tokenize text
        let prompt_ids = tokenizer
            .encode(self.text.as_str(), true)
            .map_err(|e| VoiceError::GenerationError(format!("Tokenization failed: {e}")))?
            .get_ids()
            .to_vec();
        let prompt_ids = candle_core::Tensor::new(prompt_ids, &device)
            .map_err(|e| VoiceError::GenerationError(format!("Tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| VoiceError::GenerationError(format!("Tensor reshape failed: {e}")))?;

        // Get audio prompt from speaker's voice clone if available
        let audio_prompt_codes: Option<candle_core::Tensor> =
            if let Some(voice_clone) = self.speaker.voice_clone() {
                match voice_clone.get_audio_prompt() {
                    Ok(prompt) => Some(prompt),
                    Err(e) => {
                        eprintln!("Warning: Failed to get audio prompt: {e}");
                        None
                    }
                }
            } else {
                None
            };

        // Encoder forward pass
        let prompt_b2s = prompt_ids
            .expand(&[2, prompt_ids.dim(1)?])
            .map_err(|e| VoiceError::GenerationError(format!("Prompt expansion failed: {e}")))?;
        let (enc_out, enc_state) = dia
            .encode(&prompt_b2s)
            .map_err(|e| VoiceError::GenerationError(format!("Encoding failed: {e}")))?;
        let cross_cache = dia
            .build_cross_cache(&enc_out, &enc_state.positions)
            .map_err(|e| VoiceError::GenerationError(format!("Cross cache failed: {e}")))?;

        // Prepare decoder state
        let mut dec_state = dia
            .new_decoder_state(&enc_state, enc_out, cross_cache, &device)
            .map_err(|e| VoiceError::GenerationError(format!("Decoder state failed: {e}")))?;

        // Pre-fill with audio prompt if available
        if let Some(ap) = &audio_prompt_codes {
            // Apply temporal delays before prefilling decoder
            let ap_delayed = channel_delay::delayed_view(ap, cfg.data.audio_pad_value)
                .map_err(|e| {
                    VoiceError::GenerationError(format!("Channel delay adjustment failed: {e}"))
                })?;

            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            let ap_delayed =
                channel_delay::delayed_view(&ap, cfg.data.audio_pad_value).map_err(|e| {
                    VoiceError::GenerationError(format!("Channel delay adjustment failed: {}", e))
                })?;

            let ap_b2tc = ap_delayed
                .unsqueeze(0)
                .map_err(|e| {
                    VoiceError::GenerationError(format!("Audio prompt reshape failed: {e}"))
                })?
                .expand(&[2, ap.dim(0)?, ap.dim(1)?])
                .map_err(|e| {
                    VoiceError::GenerationError(format!("Audio prompt expansion failed: {e}"))
                })?;
            dia.prefill_decoder(&ap_b2tc, &mut dec_state)
                .map_err(|e| VoiceError::GenerationError(format!("Decoder prefill failed: {e}")))?;
        }

        // Autoregressive sampling
        let mut sampler = LogitsProcessor::new(42, Some(1.0), None);
        let mut codes = Vec::<u32>::new();

        for step in 0..cfg.data.audio_length {
            dec_state.prepare_step(step);

            let toks = if step == 0 && audio_prompt_codes.is_none() {
                candle_core::Tensor::zeros(&[2, 1, cfg.data.channels], DType::U32, &device)
                    .map_err(|e| {
                        VoiceError::GenerationError(format!("BOS token creation failed: {e}"))
                    })?
            } else {
                let pad = cfg.data.audio_pad_value;
                let mut tmp = Vec::with_capacity(cfg.data.channels);
                tmp.extend_from_slice(&codes);
                tmp.push(pad);
                candle_core::Tensor::new(tmp, &device)
                    .map_err(|e| {
                        VoiceError::GenerationError(format!("Token tensor creation failed: {e}"))
                    })?
                    .reshape((2, 1, cfg.data.channels))
                    .map_err(|e| {
                        VoiceError::GenerationError(format!("Token tensor reshape failed: {e}"))
                    })?
            };

            // Apply temporal delays before model sees the tokens
            let toks = channel_delay::delayed_view(&toks, cfg.data.audio_pad_value).map_err(|e| {
                VoiceError::GenerationError(format!("Channel delay adjustment failed: {e}"))
            })?;

            let logits = dia
                .decode_step(&toks, &mut dec_state)
                .map_err(|e| VoiceError::GenerationError(format!("Decode step failed: {e}")))?;
            let logits = logits
                .i((1, .., ..))
                .map_err(|e| VoiceError::GenerationError(format!("Logits indexing failed: {e}")))?
                .to_dtype(DType::F32)
                .map_err(|e| {
                    VoiceError::GenerationError(format!("Logits dtype conversion failed: {e}"))
                })?;
            let next = sampler
                .sample(&logits)
                .map_err(|e| VoiceError::GenerationError(format!("Sampling failed: {e}")))?;

            if next == cfg.data.audio_eos_value {
                break;
            }
            codes.push(next);
        }

        // Decode audio with EnCodec
        let codes_t =
            candle_core::Tensor::from_slice(&codes, (codes.len(), cfg.data.channels), &device)
                .map_err(|e| {
                    VoiceError::GenerationError(format!("Codes tensor creation failed: {e}"))
                })?
                .unsqueeze(0)
                .map_err(|e| {
                    VoiceError::GenerationError(format!("Codes tensor reshape failed: {e}"))
                })?;

        // Remove temporal delays before EnCodec decoding
        let codes_t = channel_delay::undelayed_view(&codes_t, cfg.data.audio_pad_value).map_err(|e| {
            VoiceError::GenerationError(format!("Channel delay adjustment failed: {e}"))
        })?;

        let pcm = dia
            .decode_audio_codes(&codes_t)
            .map_err(|e| VoiceError::GenerationError(format!("Audio decoding failed: {e}")))?
            .squeeze(0)
            .map_err(|e| VoiceError::GenerationError(format!("PCM squeeze failed: {e}")))?
            .squeeze(0)
            .map_err(|e| VoiceError::GenerationError(format!("PCM squeeze failed: {e}")))?;

        // Apply loudness normalization
        let pcm = normalize_loudness(&pcm, SAMPLE_RATE as u32, true).map_err(|e| {
            VoiceError::GenerationError(format!("Loudness normalization failed: {e}"))
        })?;
        let pcm_vec = pcm
            .to_vec1::<f32>()
            .map_err(|e| VoiceError::GenerationError(format!("PCM conversion failed: {e}")))?;

        // Convert to bytes for VoicePlayer
        let mut audio_bytes = Vec::with_capacity(pcm_vec.len() * 2);
        for sample in pcm_vec {
            let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            audio_bytes.extend_from_slice(&sample_i16.to_le_bytes());
        }

        // Create VoicePlayer with the generated audio
        Ok(super::VoicePlayer::new(audio_bytes, SAMPLE_RATE as u32, 1))
    }

    /// Get the text content
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Get the speaker ID
    pub fn speaker_id(&self) -> &str {
        self.speaker.id()
    }

    /// Terminal method - execute the conversation and handle results
    pub fn player<F, T>(self, handler: F) -> T
    where
        F: FnOnce(Self) -> T,
    {
        handler(self)
    }

    /// Play the conversation with Result matching
    pub async fn play<F, T>(self, handler: F) -> T
    where
        F: FnOnce(Result<VoicePlayer, VoiceError>) -> T,
    {
        match self.internal_generate().await {
            Ok(player) => handler(Ok(player)),
            Err(error) => handler(Err(error)),
        }
    }
}
