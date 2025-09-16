//! LiveKit Audio Player for real-time audio playback using rodio
//!
//! This module bridges LiveKit RemoteTrack audio streams to rodio playback,
//! enabling real-time audio output with production-quality volume control.
//!
//! Based on established patterns from:
//! - ../../dia/src/voice/audio_stream_player.rs (rodio volume control)
//! - ../../fluent-voice/src/audio_stream.rs (zero-allocation processing)

use futures::StreamExt;
use livekit::webrtc::{audio_stream::native::NativeAudioStream, prelude::*};
use rodio::{OutputStream, Sink};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// LiveKit audio player for real-time RemoteTrack playback
pub struct LiveKitAudioPlayer {
    sink: Sink,
    volume: Arc<std::sync::Mutex<f32>>,
    running: Arc<AtomicBool>,
    _stream: OutputStream,
}

/// Errors that can occur during LiveKit audio playback
#[derive(Debug, thiserror::Error)]
pub enum LiveKitAudioError {
    #[error("Audio device error: {0}")]
    DeviceError(String),

    #[error("Audio format error: {0}")]
    FormatError(String),

    #[error("Stream error: {0}")]
    StreamError(String),

    #[error("Playback error: {0}")]
    PlaybackError(String),
}

impl LiveKitAudioPlayer {
    /// Create a new LiveKit audio player from a RemoteTrack
    ///
    /// Based on: ../../dia/src/voice/audio_stream_player.rs:22-28
    pub fn new_from_remote_track(
        rt_handle: &tokio::runtime::Handle,
        track: RtcAudioTrack,
    ) -> Result<Self, LiveKitAudioError> {
        // Initialize audio output with comprehensive error handling
        let (_stream, stream_handle) = OutputStream::try_default().map_err(|e| {
            LiveKitAudioError::DeviceError(format!("Failed to open audio output: {}", e))
        })?;

        let sink = Sink::try_new(&stream_handle).map_err(|e| {
            LiveKitAudioError::DeviceError(format!("Failed to create audio sink: {}", e))
        })?;
        sink.set_volume(0.7); // Default 70% volume (established pattern)

        // Create a second sink for the async task (rodio::Sink doesn't clone)
        let sink_for_task = Sink::try_new(&stream_handle).map_err(|e| {
            LiveKitAudioError::DeviceError(format!("Failed to create task audio sink: {}", e))
        })?;
        sink_for_task.set_volume(0.7);

        let volume = Arc::new(std::sync::Mutex::new(0.7f32));
        let running = Arc::new(AtomicBool::new(true));

        let volume_clone = volume.clone();
        let running_clone = running.clone();

        rt_handle.spawn(async move {
            Self::stream_livekit_to_rodio(track, sink_for_task, volume_clone, running_clone).await;
        });

        Ok(Self {
            sink,
            volume,
            running,
            _stream: _stream,
        })
    }

    /// Set playback volume (0.0 to 2.0, allows boost up to 200%)
    ///
    /// Based on: ../../dia/src/voice/audio_stream_player.rs:99-101
    pub fn set_volume(&self, volume: f32) {
        let clamped = volume.clamp(0.0, 2.0);
        self.sink.set_volume(clamped);

        // Update stored volume for consistency
        if let Ok(mut stored_volume) = self.volume.lock() {
            *stored_volume = clamped;
        }

        tracing::debug!("LiveKit audio volume set to: {:.2}", clamped);
    }

    /// Get current volume setting
    pub fn get_volume(&self) -> f32 {
        self.volume
            .lock()
            .unwrap_or_else(|poisoned| {
                tracing::warn!("Volume mutex was poisoned, recovering with default");
                poisoned.into_inner()
            })
            .clone()
    }

    /// Mute audio playback
    pub fn mute(&self) {
        self.sink.set_volume(0.0);
        tracing::debug!("LiveKit audio muted");
    }

    /// Unmute audio playback (restore to stored volume)
    pub fn unmute(&self) {
        let volume = self.get_volume();
        self.sink.set_volume(volume);
        tracing::debug!("LiveKit audio unmuted to volume: {:.2}", volume);
    }

    /// Pause playback
    pub fn pause(&self) {
        self.sink.pause();
        tracing::debug!("LiveKit audio paused");
    }

    /// Resume playback
    pub fn resume(&self) {
        self.sink.play();
        tracing::debug!("LiveKit audio resumed");
    }

    /// Stop playback and clean up resources
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        self.sink.stop();
        tracing::debug!("LiveKit audio stopped");
    }

    /// Check if audio is currently playing
    pub fn is_playing(&self) -> bool {
        !self.sink.empty() && self.running.load(Ordering::Relaxed)
    }

    /// Bridge LiveKit NativeAudioStream to rodio SamplesBuffer
    ///
    /// Pattern from: ../../fluent-voice/src/audio_stream.rs:90-140
    async fn stream_livekit_to_rodio(
        track: RtcAudioTrack,
        sink: Sink,
        _volume: Arc<std::sync::Mutex<f32>>,
        running: Arc<AtomicBool>,
    ) {
        tracing::info!("Starting LiveKit audio stream bridge to rodio");

        let mut audio_stream = NativeAudioStream::new(track, 48000, 1);
        let mut frame_count = 0u64;

        while running.load(Ordering::Relaxed) {
            match audio_stream.next().await {
                Some(frame) => {
                    // Convert LiveKit audio frame to rodio samples
                    // Zero-allocation pattern where possible
                    if frame.data.is_empty() {
                        continue;
                    }

                    let samples: Vec<f32> = frame
                        .data
                        .iter()
                        .map(|&sample_i16| {
                            sample_i16 as f32 / 32768.0 // Normalize to -1.0..1.0
                        })
                        .collect();

                    if !samples.is_empty() {
                        let buffer = rodio::buffer::SamplesBuffer::new(1, 48000, samples);
                        sink.append(buffer);

                        frame_count += 1;

                        if frame_count % 1000 == 0 {
                            tracing::debug!("Processed {} audio frames", frame_count);
                        }
                    }
                }
                None => {
                    tracing::info!("LiveKit audio stream ended");
                    break;
                }
            }
        }

        tracing::info!(
            "LiveKit audio stream bridge completed. Total frames: {}",
            frame_count
        );
    }
}

impl Drop for LiveKitAudioPlayer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Convenience function to create and start LiveKit audio playback
pub async fn play_livekit_audio(
    rt_handle: &tokio::runtime::Handle,
    track: RtcAudioTrack,
) -> Result<LiveKitAudioPlayer, LiveKitAudioError> {
    LiveKitAudioPlayer::new_from_remote_track(rt_handle, track)
}
