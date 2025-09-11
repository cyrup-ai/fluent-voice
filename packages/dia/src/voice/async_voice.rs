//! AsyncVoice trait - futures that resolve to VoicePlayer

use super::VoicePlayer;
use crate::config::DiaConfig;
use crate::model::{DiaModel, load_dia_model};
use crate::setup::ModelPaths;
use crate::state::DecoderInferenceState;
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

/// Trait for voice generators that can produce audio asynchronously
/// Returns synchronous interfaces that provide awaitable results
pub trait AsyncVoice: Send + Sync {
    /// Generate voice audio and return a task that resolves to VoicePlayer
    fn generate(&self) -> VoiceGenerationTask;
}

/// A task that asynchronously generates voice audio
pub struct VoiceGenerationTask {
    text: String,
    model_paths: ModelPaths,
    device: Device,
}

impl VoiceGenerationTask {
    pub fn new(text: String, model_paths: ModelPaths, device: Device) -> Self {
        Self {
            text,
            model_paths,
            device,
        }
    }

    /// Await the voice generation result
    pub async fn await_result(self) -> Result<VoicePlayer, VoiceError> {
        tracing::info!("Starting Dia voice synthesis for: {}", self.text);

        // Load tokenizer for text processing
        let tokenizer = self.load_tokenizer().await?;

        // Tokenize input text
        let encoding = tokenizer
            .encode(self.text.as_str(), true)
            .map_err(|e| VoiceError::GenerationError(format!("Tokenization failed: {e}")))?;
        let tokens = encoding.get_ids();

        // Convert tokens to tensor
        let input_tensor = Tensor::new(tokens, &self.device).map_err(|e| {
            VoiceError::GenerationError(format!("Failed to create input tensor: {e}"))
        })?;

        // Load Dia TTS model using existing infrastructure
        let config = DiaConfig::default();
        let dia_model = load_dia_model(&self.model_paths.weights, &config, &self.device)?;

        // Encode text input
        let (enc_out, enc_state) = dia_model.encode(&input_tensor)?;

        // Build cross-attention cache
        let cross_cache = dia_model.build_cross_cache(&enc_out, &input_tensor)?;

        // Create decoder state
        let mut dec_state =
            dia_model.new_decoder_state(&enc_state, enc_out, cross_cache, &self.device)?;

        // Generate audio codes through autoregressive decoding
        let audio_codes = self
            .generate_audio_codes(&dia_model, &mut dec_state)
            .await?;

        // Decode audio codes to waveform using EnCodec
        let audio_tensor = dia_model.decode_audio_codes(&audio_codes)?;

        // Convert tensor to audio bytes
        let audio_data = self.tensor_to_audio_bytes(&audio_tensor)?;

        tracing::info!(
            "Dia voice synthesis completed: {} samples generated",
            audio_data.len() / 2 // 16-bit samples
        );

        // Create VoicePlayer with synthesized audio
        let voice_player = VoicePlayer::new(audio_data, 24000, 1);
        Ok(voice_player)
    }

    /// Load tokenizer from model paths
    async fn load_tokenizer(&self) -> Result<Tokenizer, VoiceError> {
        let tokenizer_path = &self.model_paths.tokenizer;

        if !tokenizer_path.exists() {
            return Err(VoiceError::ConfigError(format!(
                "Tokenizer file not found: {}",
                tokenizer_path.display()
            )));
        }

        Tokenizer::from_file(tokenizer_path)
            .map_err(|e| VoiceError::GenerationError(format!("Failed to load tokenizer: {e}")))
    }

    /// Generate audio codes through autoregressive decoding
    async fn generate_audio_codes(
        &self,
        model: &DiaModel,
        dec_state: &mut DecoderInferenceState,
    ) -> Result<Tensor, VoiceError> {
        // Start with silence token (assuming token 0 is silence)
        let mut current_codes = vec![0u32; 8]; // 8 codebooks for EnCodec
        let mut all_codes = Vec::new();

        // Generate audio codes autoregressively
        for _step in 0..1000 {
            // Max 1000 steps for reasonable audio length
            // Convert current codes to tensor
            let codes_tensor = Tensor::new(&current_codes[..], &self.device)
                .map_err(|e| {
                    VoiceError::GenerationError(format!("Failed to create codes tensor: {e}"))
                })?
                .unsqueeze(0)? // Add batch dimension
                .unsqueeze(0)?; // Add sequence dimension

            // Decode step to get next codes
            let logits = model.decode_step(&codes_tensor, dec_state)?;

            // Sample from logits (using greedy decoding for simplicity)
            let next_codes = self.sample_from_logits(&logits)?;

            // Check for end-of-sequence
            if next_codes.iter().all(|&code| code == 0) {
                break;
            }

            all_codes.extend_from_slice(&next_codes);
            current_codes = next_codes;
        }

        // Convert to tensor with proper shape for EnCodec
        let codes_len = all_codes.len() / 8; // 8 codebooks
        let codes_tensor = Tensor::new(all_codes, &self.device)?.reshape((1, 8, codes_len))?; // [batch, codebooks, time]

        Ok(codes_tensor)
    }

    /// Sample from logits tensor (greedy decoding)
    fn sample_from_logits(&self, logits: &Tensor) -> Result<Vec<u32>, VoiceError> {
        use candle_core::IndexOp;
        // Get the last timestep logits
        let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;

        // Apply argmax to get most likely tokens
        let codes = last_logits.argmax_keepdim(2)?;

        // Convert to Vec<u32>
        let codes_vec = codes
            .to_vec2::<u32>()
            .map_err(|e| VoiceError::GenerationError(format!("Failed to extract codes: {e}")))?;

        Ok(codes_vec[0].clone())
    }

    /// Convert audio tensor to bytes
    fn tensor_to_audio_bytes(&self, audio_tensor: &Tensor) -> Result<Vec<u8>, VoiceError> {
        // Extract audio data as f32 samples
        let audio_data = audio_tensor.to_vec1::<f32>().map_err(|e| {
            VoiceError::GenerationError(format!("Failed to extract audio data: {e}"))
        })?;

        // Convert f32 samples to i16 bytes
        let mut audio_bytes = Vec::with_capacity(audio_data.len() * 2);
        for sample in audio_data {
            let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            audio_bytes.extend_from_slice(&sample_i16.to_le_bytes());
        }

        Ok(audio_bytes)
    }
}

/// Errors that can occur during voice synthesis
#[derive(Debug, thiserror::Error)]
pub enum VoiceError {
    #[error("Failed to load voice clone: {0}")]
    CloneLoadError(String),

    #[error("Audio generation failed: {0}")]
    GenerationError(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("No default speaker configured")]
    NoDefaultSpeaker,
}

impl From<candle_core::Error> for VoiceError {
    fn from(err: candle_core::Error) -> Self {
        VoiceError::GenerationError(format!("Candle error: {err}"))
    }
}

impl From<crate::model::ModelError> for VoiceError {
    fn from(err: crate::model::ModelError) -> Self {
        VoiceError::GenerationError(format!("Model error: {err}"))
    }
}
