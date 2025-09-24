//! Audio Stream Manager for CPAL Integration
//!
//! This module provides a thread-safe interface for CPAL audio streams by separating
//! the non-Send CPAL stream from the Send conversation context using crossbeam channels.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender};
use fluent_voice_domain::VoiceError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Configuration for audio stream setup
#[derive(Debug, Clone)]
pub struct AudioStreamConfig {
    /// Sample rate in Hz (typically 16000 for speech)
    pub sample_rate: u32,
    /// Number of audio channels (typically 1 for mono)
    pub channels: u16,
    /// Device name for audio capture (empty string for default device)
    pub device_name: String,
}

impl Default for AudioStreamConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            device_name: String::new(),
        }
    }
}

/// Audio Stream Manager that handles CPAL streams in a separate thread
///
/// This architecture separates the non-Send CPAL stream components from the Send
/// conversation logic by running the audio capture in its own thread and communicating
/// via crossbeam channels.
pub struct AudioStreamManager {
    _phantom_sender: Option<Sender<Vec<f32>>>,
    shutdown_flag: Arc<AtomicBool>,
    thread_handle: Option<JoinHandle<Result<(), VoiceError>>>,
}

impl AudioStreamManager {
    /// Create a new AudioStreamManager with the specified configuration
    ///
    /// Returns the manager and a receiver for audio data chunks
    pub fn new(config: AudioStreamConfig) -> Result<(Self, Receiver<Vec<f32>>), VoiceError> {
        let (audio_sender, audio_receiver) = crossbeam_channel::unbounded();
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown_flag.clone();

        // Clone sender for the thread
        let sender_clone = audio_sender.clone();

        // Spawn thread for CPAL stream management
        let thread_handle = thread::spawn(move || -> Result<(), VoiceError> {
            Self::run_audio_capture_thread(config, sender_clone, shutdown_clone)
        });

        Ok((
            Self {
                _phantom_sender: Some(audio_sender),
                shutdown_flag,
                thread_handle: Some(thread_handle),
            },
            audio_receiver,
        ))
    }

    /// Run the audio capture thread with CPAL stream
    fn run_audio_capture_thread(
        config: AudioStreamConfig,
        sender: Sender<Vec<f32>>,
        shutdown_flag: Arc<AtomicBool>,
    ) -> Result<(), VoiceError> {
        // Initialize CPAL host and device
        let host = cpal::default_host();

        let device = if config.device_name.is_empty() {
            host.default_input_device().ok_or_else(|| {
                VoiceError::Configuration("No default input device available".to_string())
            })?
        } else {
            host.input_devices()
                .map_err(|e| {
                    VoiceError::Configuration(format!("Failed to enumerate input devices: {}", e))
                })?
                .find(|device| {
                    device
                        .name()
                        .map(|name| name == config.device_name)
                        .unwrap_or(false)
                })
                .ok_or_else(|| {
                    VoiceError::Configuration(format!(
                        "Input device '{}' not found",
                        config.device_name
                    ))
                })?
        };

        // Configure stream
        let stream_config = cpal::StreamConfig {
            channels: config.channels,
            sample_rate: cpal::SampleRate(config.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        // Build input stream based on sample format
        let supported_config = device.default_input_config().map_err(|e| {
            VoiceError::Configuration(format!("Failed to get default input config: {}", e))
        })?;

        let stream = match supported_config.sample_format() {
            cpal::SampleFormat::I8 => device.build_input_stream(
                &stream_config,
                move |data: &[i8], _: &cpal::InputCallbackInfo| {
                    let float_data: Vec<f32> = data
                        .iter()
                        .map(|&sample| sample as f32 / i8::MAX as f32)
                        .collect();
                    if sender.try_send(float_data).is_err() {
                        tracing::warn!("Audio channel full, dropping audio data");
                    }
                },
                move |err| {
                    tracing::error!("Audio stream error: {}", err);
                },
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let float_data: Vec<f32> = data
                        .iter()
                        .map(|&sample| sample as f32 / i16::MAX as f32)
                        .collect();
                    if sender.try_send(float_data).is_err() {
                        tracing::warn!("Audio channel full, dropping audio data");
                    }
                },
                move |err| {
                    tracing::error!("Audio stream error: {}", err);
                },
                None,
            ),
            cpal::SampleFormat::I32 => device.build_input_stream(
                &stream_config,
                move |data: &[i32], _: &cpal::InputCallbackInfo| {
                    let float_data: Vec<f32> = data
                        .iter()
                        .map(|&sample| sample as f32 / i32::MAX as f32)
                        .collect();
                    if sender.try_send(float_data).is_err() {
                        tracing::warn!("Audio channel full, dropping audio data");
                    }
                },
                move |err| {
                    tracing::error!("Audio stream error: {}", err);
                },
                None,
            ),
            cpal::SampleFormat::U8 => device.build_input_stream(
                &stream_config,
                move |data: &[u8], _: &cpal::InputCallbackInfo| {
                    let float_data: Vec<f32> = data
                        .iter()
                        .map(|&sample| (sample as f32 - 128.0) / 128.0)
                        .collect();
                    if sender.try_send(float_data).is_err() {
                        tracing::warn!("Audio channel full, dropping audio data");
                    }
                },
                move |err| {
                    tracing::error!("Audio stream error: {}", err);
                },
                None,
            ),
            cpal::SampleFormat::U16 => device.build_input_stream(
                &stream_config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let float_data: Vec<f32> = data
                        .iter()
                        .map(|&sample| (sample as f32 - 32768.0) / 32768.0)
                        .collect();
                    if sender.try_send(float_data).is_err() {
                        tracing::warn!("Audio channel full, dropping audio data");
                    }
                },
                move |err| {
                    tracing::error!("Audio stream error: {}", err);
                },
                None,
            ),
            cpal::SampleFormat::U32 => device.build_input_stream(
                &stream_config,
                move |data: &[u32], _: &cpal::InputCallbackInfo| {
                    let float_data: Vec<f32> = data
                        .iter()
                        .map(|&sample| (sample as f32 - 2147483648.0) / 2147483648.0)
                        .collect();
                    if sender.try_send(float_data).is_err() {
                        tracing::warn!("Audio channel full, dropping audio data");
                    }
                },
                move |err| {
                    tracing::error!("Audio stream error: {}", err);
                },
                None,
            ),
            cpal::SampleFormat::F32 => device.build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if sender.try_send(data.to_vec()).is_err() {
                        tracing::warn!("Audio channel full, dropping audio data");
                    }
                },
                move |err| {
                    tracing::error!("Audio stream error: {}", err);
                },
                None,
            ),
            cpal::SampleFormat::F64 => device.build_input_stream(
                &stream_config,
                move |data: &[f64], _: &cpal::InputCallbackInfo| {
                    let float_data: Vec<f32> = data.iter().map(|&sample| sample as f32).collect();
                    if sender.try_send(float_data).is_err() {
                        tracing::warn!("Audio channel full, dropping audio data");
                    }
                },
                move |err| {
                    tracing::error!("Audio stream error: {}", err);
                },
                None,
            ),
            sample_format => {
                return Err(VoiceError::Configuration(format!(
                    "Unsupported sample format: {:?}",
                    sample_format
                )));
            }
        }
        .map_err(|e| VoiceError::Configuration(format!("Failed to build input stream: {}", e)))?;

        // Start the stream
        stream.play().map_err(|e| {
            VoiceError::Configuration(format!("Failed to start audio stream: {}", e))
        })?;

        tracing::info!("Audio capture thread started");

        // Keep the stream alive until shutdown
        while !shutdown_flag.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        tracing::info!("Audio capture thread shutting down");
        Ok(())
    }

    /// Check if the audio stream is currently active
    pub fn is_active(&self) -> bool {
        !self.shutdown_flag.load(Ordering::Relaxed) && self.thread_handle.is_some()
    }

    /// Stop the audio stream and clean up resources
    pub fn stop(&mut self) {
        self.shutdown_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.thread_handle.take() {
            // Wait for thread to finish
            if let Err(e) = handle.join() {
                tracing::error!("Audio capture thread panicked: {:?}", e);
            }
        }
    }
}

impl Drop for AudioStreamManager {
    fn drop(&mut self) {
        self.stop();
    }
}
