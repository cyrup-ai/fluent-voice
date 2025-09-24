//! Voice parameters and speaker PCM data structures

use super::error::SpeechGenerationError;

/// Voice parameters for speech synthesis control
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VoiceParameters {
    /// Speech rate multiplier (0.5 = half speed, 2.0 = double speed)
    pub speed: f32,
    /// Pitch adjustment in semitones (-12.0 to +12.0)
    pub pitch: f32,
    /// Voice emphasis/intensity (0.0 to 2.0)
    pub emphasis: f32,
    /// Emotional tone (-1.0 = sad, 0.0 = neutral, 1.0 = happy)
    pub emotion: f32,
    /// Breathing pause duration multiplier (0.0 to 2.0)
    pub pause_duration: f32,
    /// Volume level (0.0 to 1.0)
    pub volume: f32,
    /// Path to voice clone audio file for speaker PCM processing
    pub voice_clone_path: Option<std::path::PathBuf>,
}

/// Comprehensive parameter storage for speaker PCM processing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpeakerPcmData {
    pub speaker_id: String,
    pub pcm_samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub embedding: Option<Vec<f32>>,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Configuration for speaker PCM processing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpeakerPcmConfig {
    pub target_sample_rate: u32,
    pub target_channels: u16,
    pub embedding_dim: usize,
    pub min_samples: usize,
    pub max_samples: usize,
    pub normalization_enabled: bool,
    /// Audio cache configuration - maximum cache size in MB
    pub audio_cache_max_size_mb: usize,
    /// Enable streaming processing for large files (>10MB)
    pub streaming_enabled: bool,
    /// Chunk size for streaming processing (samples)
    pub streaming_chunk_size: usize,
}

impl Default for SpeakerPcmConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 24000, // Match Moshi's expected 24kHz sample rate
            target_channels: 1,        // Mono for speaker identification
            embedding_dim: 512,        // Standard speaker embedding dimension
            min_samples: 2400,         // 100ms minimum at 24kHz
            max_samples: 240000,       // 10s maximum at 24kHz
            normalization_enabled: true,
            audio_cache_max_size_mb: 256, // 256MB cache limit
            streaming_enabled: true,      // Enable streaming for large files
            streaming_chunk_size: 44100,  // 1 second chunks at 44.1kHz
        }
    }
}

impl Default for VoiceParameters {
    fn default() -> Self {
        Self {
            speed: 1.0,
            pitch: 0.0,
            emphasis: 1.0,
            emotion: 0.0,
            pause_duration: 1.0,
            volume: 0.8,
            voice_clone_path: None,
        }
    }
}

impl VoiceParameters {
    /// Validate voice parameters are within acceptable ranges
    pub fn validate(&self) -> Result<(), SpeechGenerationError> {
        if !(0.1..=5.0).contains(&self.speed) {
            return Err(SpeechGenerationError::InvalidVoiceParameters(format!(
                "Speed must be between 0.1 and 5.0, got {}",
                self.speed
            )));
        }
        if !(-24.0..=24.0).contains(&self.pitch) {
            return Err(SpeechGenerationError::InvalidVoiceParameters(format!(
                "Pitch must be between -24.0 and 24.0 semitones, got {}",
                self.pitch
            )));
        }
        if !(0.0..=2.0).contains(&self.emphasis) {
            return Err(SpeechGenerationError::InvalidVoiceParameters(format!(
                "Emphasis must be between 0.0 and 2.0, got {}",
                self.emphasis
            )));
        }
        if !(-1.0..=1.0).contains(&self.emotion) {
            return Err(SpeechGenerationError::InvalidVoiceParameters(format!(
                "Emotion must be between -1.0 and 1.0, got {}",
                self.emotion
            )));
        }
        if !(0.0..=2.0).contains(&self.pause_duration) {
            return Err(SpeechGenerationError::InvalidVoiceParameters(format!(
                "Pause duration must be between 0.0 and 2.0, got {}",
                self.pause_duration
            )));
        }
        if !(0.0..=1.0).contains(&self.volume) {
            return Err(SpeechGenerationError::InvalidVoiceParameters(format!(
                "Volume must be between 0.0 and 1.0, got {}",
                self.volume
            )));
        }
        Ok(())
    }
}
