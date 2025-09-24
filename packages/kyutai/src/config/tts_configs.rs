//! Text-to-Speech configuration types

/// Configuration for Text-to-Speech model parameters
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TtsConfig {
    pub acoustic_delay: usize,
    pub text_pad_token: u32,
    pub text_bos_token: u32,
    pub text_eos_token: u32,
    pub text_eop_token: u32,
    pub text_start_token: u32,
    pub text_audio_delay_in_tokens: usize,
    pub max_consecutive_pads: usize,
    pub speaker_cond_duration_s: f64,
    pub speaker_cond_dim: usize,
    pub speaker_cond_n_speakers: usize,
    pub second_stream_ahead: usize,
}

impl TtsConfig {
    /// Configuration for 2025 TTS models
    pub fn v202501() -> Self {
        Self {
            acoustic_delay: 2,
            text_eop_token: 0,
            text_bos_token: 1,
            text_eos_token: 2,
            text_pad_token: 3,
            text_start_token: 8000,
            text_audio_delay_in_tokens: 16,
            max_consecutive_pads: 10,
            speaker_cond_duration_s: 10.,
            speaker_cond_dim: 512,
            speaker_cond_n_speakers: 5,
            second_stream_ahead: 2,
        }
    }
}
