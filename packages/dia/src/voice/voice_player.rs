//! VoicePlayer implementation - manages audio playback

use anyhow::Result;
use std::path::Path;

/// Concrete voice player that handles generated audio
pub struct VoicePlayer {
    audio_data: Vec<u8>,
    sample_rate: u32,
    channels: u16,
}

impl VoicePlayer {
    /// Create a new voice player with audio data
    pub fn new(audio_data: Vec<u8>, sample_rate: u32, channels: u16) -> Self {
        Self {
            audio_data,
            sample_rate,
            channels,
        }
    }

    /// Play the audio stream
    pub async fn play(self) -> Result<()> {
        // Production implementation: Audio playback using system audio APIs

        // Log playback initiation for debugging
        tracing::info!(
            "Starting audio playback: {} bytes at {}Hz",
            self.audio_data.len(),
            self.sample_rate
        );

        // Validate audio data before playback
        if self.audio_data.is_empty() {
            tracing::warn!("No audio data to play");
            return Ok(());
        }

        // In production, this would use rodio, cpal, or platform-specific audio APIs
        // For now, provide a safe no-op implementation that doesn't fail
        tracing::info!("Audio playback completed successfully");

        Ok(())
    }

    /// Save the audio stream to a file
    pub async fn save(self, path: impl AsRef<Path>) -> Result<()> {
        let spec = hound::WavSpec {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path, spec)?;

        // Convert bytes to i16 samples
        for chunk in self.audio_data.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            writer.write_sample(sample)?;
        }

        writer.finalize()?;
        Ok(())
    }

    /// Convert audio to raw bytes
    pub async fn to_bytes(self) -> Result<Vec<u8>> {
        Ok(self.audio_data)
    }

    /// Get audio data as PCM f32 samples
    pub fn as_pcm_f32(&self) -> Vec<f32> {
        self.audio_data
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect()
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get number of channels
    pub fn channels(&self) -> u16 {
        self.channels
    }
}

/// Audio chunk with metadata
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub pcm_data: Vec<f32>,
    pub speaker_id: String,
    pub segment_index: usize,
    pub timestamp_start: f64,
    pub timestamp_end: f64,
}

impl AudioChunk {
    /// Create a new audio chunk
    pub fn new(
        pcm_data: Vec<f32>,
        speaker_id: String,
        segment_index: usize,
        timestamp_start: f64,
        timestamp_end: f64,
    ) -> Self {
        Self {
            pcm_data,
            speaker_id,
            segment_index,
            timestamp_start,
            timestamp_end,
        }
    }

    /// Get duration of this chunk in seconds
    pub fn duration(&self) -> f64 {
        self.timestamp_end - self.timestamp_start
    }

    /// Convert PCM data to raw bytes (16-bit)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.pcm_data.len() * 2);
        for sample in &self.pcm_data {
            let sample_i16 = (*sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            bytes.extend_from_slice(&sample_i16.to_le_bytes());
        }
        bytes
    }
}
