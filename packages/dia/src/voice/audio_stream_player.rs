//! Audio Stream Player for real-time AudioChunk playback using rodio
//!
//! This module provides streaming audio playback for AudioChunk streams,
//! enabling real-time TTS audio output with production-quality rodio integration.

use crate::voice::tts_builder::AudioChunk;
use futures_core::Stream;
use futures_util::StreamExt;
use rodio::{OutputStream, OutputStreamBuilder, Sink, Source};
use std::time::Duration;

/// Audio stream player for real-time AudioChunk playback
pub struct AudioStreamPlayer {
    _stream: OutputStream,
    sink: Sink,
}

impl AudioStreamPlayer {
    /// Create a new audio stream player
    pub fn new() -> Result<Self, AudioStreamError> {
        let stream = OutputStreamBuilder::open_default_stream().map_err(|e| {
            AudioStreamError::DeviceError(format!("Failed to open audio output: {}", e))
        })?;

        let sink = Sink::connect_new(stream.mixer());

        // Set reasonable volume (70%)
        sink.set_volume(0.7);

        Ok(Self {
            _stream: stream,
            sink,
        })
    }

    /// Play an AudioChunk stream with real-time audio output
    pub async fn play_stream<S>(self, mut stream: S) -> Result<(), AudioStreamError>
    where
        S: Stream<Item = AudioChunk> + Send + Unpin + 'static,
    {
        tracing::info!("Starting real-time AudioChunk stream playback");

        let mut chunk_count = 0u64;
        let mut total_bytes = 0usize;

        while let Some(chunk) = stream.next().await {
            // Check for error chunks
            if let Some(error_msg) = chunk.metadata.get("error") {
                tracing::error!("Audio stream error: {}", error_msg);
                return Err(AudioStreamError::StreamError(error_msg.to_string()));
            }

            // Skip empty chunks
            if chunk.audio_data.is_empty() {
                if chunk.is_final {
                    tracing::info!("Received final chunk, completing playback");
                    break;
                }
                continue;
            }

            // Convert AudioChunk to rodio source
            let audio_source = AudioChunkSource::new(chunk.clone())?;

            // Append to sink for continuous playback
            self.sink.append(audio_source);

            chunk_count += 1;
            total_bytes += chunk.audio_data.len();

            tracing::debug!(
                "Played audio chunk {}: {} bytes (total: {} bytes)",
                chunk_count,
                chunk.audio_data.len(),
                total_bytes
            );

            // Check if this is the final chunk
            if chunk.is_final {
                tracing::info!("Received final chunk, completing playback");
                break;
            }
        }

        // Wait for all audio to finish playing using async approach
        while !self.sink.empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        tracing::info!(
            "AudioChunk stream playback completed: {} chunks, {} total bytes",
            chunk_count,
            total_bytes
        );

        Ok(())
    }

    /// Set playback volume (0.0 to 1.0)
    pub fn set_volume(&self, volume: f32) {
        self.sink.set_volume(volume.clamp(0.0, 1.0));
    }

    /// Pause playback
    pub fn pause(&self) {
        self.sink.pause();
    }

    /// Resume playback
    pub fn resume(&self) {
        self.sink.play();
    }

    /// Stop playback immediately
    pub fn stop(&self) {
        self.sink.stop();
    }
}

/// Rodio Source implementation for AudioChunk
struct AudioChunkSource {
    data: Vec<i16>,
    sample_rate: u32,
    channels: u16,
    position: usize,
}

impl AudioChunkSource {
    fn new(chunk: AudioChunk) -> Result<Self, AudioStreamError> {
        // Convert bytes to i16 samples
        if !chunk.audio_data.len().is_multiple_of(2) {
            return Err(AudioStreamError::FormatError(
                "Audio data length must be even for i16 samples".to_string(),
            ));
        }

        let mut data = Vec::with_capacity(chunk.audio_data.len() / 2);
        for chunk_bytes in chunk.audio_data.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk_bytes[0], chunk_bytes[1]]);
            data.push(sample);
        }

        let sample_rate = chunk.sample_rate.unwrap_or(24000);
        let channels = chunk
            .metadata
            .get("channels")
            .and_then(|v| v.parse::<u64>().ok())
            .map(|v| v as u16)
            .unwrap_or(1);

        Ok(Self {
            data,
            sample_rate,
            channels,
            position: 0,
        })
    }
}

impl Iterator for AudioChunkSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.data.len() {
            let sample = self.data[self.position];
            self.position += 1;
            // Convert i16 to f32 normalized to -1.0..1.0 range
            Some(sample as f32 / 32768.0)
        } else {
            None
        }
    }
}

impl Source for AudioChunkSource {
    fn current_span_len(&self) -> Option<usize> {
        Some(self.data.len() - self.position)
    }

    fn channels(&self) -> rodio::ChannelCount {
        self.channels
    }

    fn sample_rate(&self) -> rodio::SampleRate {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        let samples = self.data.len() as u64;
        let duration_secs = samples / (self.sample_rate as u64 * self.channels as u64);
        let duration_nanos = ((samples % (self.sample_rate as u64 * self.channels as u64))
            * 1_000_000_000)
            / (self.sample_rate as u64 * self.channels as u64);
        Some(Duration::new(duration_secs, duration_nanos as u32))
    }
}

/// Convenience function to play an AudioChunk stream
pub async fn play_audio_stream<S>(stream: S) -> Result<(), AudioStreamError>
where
    S: Stream<Item = AudioChunk> + Send + Unpin + 'static,
{
    let player = AudioStreamPlayer::new()?;
    player.play_stream(stream).await
}

/// Errors that can occur during audio stream playback
#[derive(Debug, thiserror::Error)]
pub enum AudioStreamError {
    #[error("Audio device error: {0}")]
    DeviceError(String),

    #[error("Audio format error: {0}")]
    FormatError(String),

    #[error("Stream error: {0}")]
    StreamError(String),

    #[error("Playback error: {0}")]
    PlaybackError(String),
}
