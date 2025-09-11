//! VoicePlayer implementation - manages audio playback

use anyhow::Result;
use std::path::Path;

/// Concrete voice player that handles generated audio
pub struct VoicePlayer {
    pub audio_data: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u16,
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

        // Real rodio-based audio playback implementation
        use rodio::{OutputStreamBuilder, Sink};

        // Create audio output stream using correct API
        let stream_handle = OutputStreamBuilder::open_default_stream()
            .map_err(|e| anyhow::anyhow!("Failed to create audio output stream: {}", e))?;

        // Create audio sink for playback
        let sink = Sink::connect_new(stream_handle.mixer());

        // Convert audio bytes to cursor for rodio
        let _cursor = std::io::Cursor::new(self.audio_data.clone());

        // Create WAV decoder from raw PCM data
        let spec = hound::WavSpec {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        // Create temporary WAV file in memory
        let mut wav_data = Vec::new();
        {
            let mut writer = hound::WavWriter::new(std::io::Cursor::new(&mut wav_data), spec)
                .map_err(|e| anyhow::anyhow!("Failed to create WAV writer: {}", e))?;

            // Write PCM samples to WAV format - use explicit slice operations to avoid sized_chunks conflicts
            let mut i = 0;
            while i + 1 < self.audio_data.len() {
                let sample = i16::from_le_bytes([self.audio_data[i], self.audio_data[i + 1]]);
                writer
                    .write_sample(sample)
                    .map_err(|e| anyhow::anyhow!("Failed to write audio sample: {}", e))?;
                i += 2;
            }

            writer
                .finalize()
                .map_err(|e| anyhow::anyhow!("Failed to finalize WAV data: {}", e))?;
        }

        // Create decoder from WAV data
        let wav_cursor = std::io::Cursor::new(wav_data);
        let decoder = rodio::Decoder::new(wav_cursor)
            .map_err(|e| anyhow::anyhow!("Failed to create audio decoder: {}", e))?;

        // Play audio through sink
        sink.append(decoder);

        // Wait for playback to complete using async approach
        while !sink.empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

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

        // Convert bytes to i16 samples - use explicit slice operations to avoid sized_chunks conflicts
        let mut i = 0;
        while i + 1 < self.audio_data.len() {
            let sample = i16::from_le_bytes([self.audio_data[i], self.audio_data[i + 1]]);
            writer.write_sample(sample)?;
            i += 2;
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
        let mut result = Vec::with_capacity(self.audio_data.len() / 2);
        let mut i = 0;
        while i + 1 < self.audio_data.len() {
            let sample = i16::from_le_bytes([self.audio_data[i], self.audio_data[i + 1]]);
            result.push(sample as f32 / 32768.0);
            i += 2;
        }
        result
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
