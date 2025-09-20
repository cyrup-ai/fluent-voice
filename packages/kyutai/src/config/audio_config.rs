//! Audio processing configuration types

use serde::{Deserialize, Serialize};

/// Audio processing configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AudioConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u32,
    /// Frame size for processing
    pub frame_size: usize,
    /// Hop length for windowing
    pub hop_length: usize,
    /// Number of mel frequency bins
    pub n_mels: usize,
    /// FFT window size
    pub n_fft: usize,
    /// Minimum frequency for mel scale
    pub f_min: f32,
    /// Maximum frequency for mel scale
    pub f_max: Option<f32>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            channels: 1,
            frame_size: 1024,
            hop_length: 256,
            n_mels: 80,
            n_fft: 1024,
            f_min: 0.0,
            f_max: None,
        }
    }
}
