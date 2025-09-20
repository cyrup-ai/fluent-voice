//! Configuration types for speech generation

use super::voice_params::{SpeakerPcmConfig, VoiceParameters};
use crate::tts::Config as TtsConfig;
use candle_core::{DType, Device};

/// Audio buffer size for streaming generation (16KB = ~180ms at 44.1kHz)
const AUDIO_BUFFER_SIZE: usize = 16384;

/// Configuration for speech generation engine
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// TTS model configuration
    pub tts_config: TtsConfig,
    /// Voice parameters
    pub voice_params: VoiceParameters,
    /// Maximum generation steps
    pub max_steps: usize,
    /// Generation temperature (0.0 to 2.0)
    pub temperature: f64,
    /// Top-k sampling parameter
    pub top_k: usize,
    /// Top-p nucleus sampling parameter
    pub top_p: f64,
    /// Random seed for reproducible generation
    pub seed: u64,
    /// Enable real-time streaming
    pub enable_streaming: bool,
    /// Audio buffer size for streaming
    pub stream_buffer_size: usize,
    /// Device for computation
    pub device: Device,
    /// Data type for tensors
    pub dtype: DType,
    /// Speaker PCM processing configuration
    pub speaker_pcm: SpeakerPcmConfig,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            tts_config: TtsConfig::v202501(),
            voice_params: VoiceParameters::default(),
            max_steps: 2000,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            seed: 42,
            enable_streaming: true,
            stream_buffer_size: AUDIO_BUFFER_SIZE,
            device: Device::Cpu,
            dtype: DType::F32,
            speaker_pcm: SpeakerPcmConfig::default(),
        }
    }
}
