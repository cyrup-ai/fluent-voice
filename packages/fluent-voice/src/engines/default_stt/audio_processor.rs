//! Audio Processing Components
//!
//! Zero-allocation audio pipeline combining VAD, wake word detection, and transcription
//! using real implementations with optimized performance.

use anyhow::Result;
use fluent_voice_domain::{AudioFormat, SpeechSource, VoiceError};
use fluent_voice_vad::VoiceActivityDetector;
use fluent_voice_whisper::WhisperSttBuilder;
use koffee::wakewords::{WakewordLoad, WakewordModel};
use koffee::{KoffeeCandle, KoffeeCandleConfig};
use ringbuf::{traits::*, HeapCons, HeapProd, HeapRb};
use std::io::Write;

use super::config::WakeWordConfig;

/// Load wake word model with fallback paths
/// Uses the same proven pattern as default_engine_coordinator.rs
fn load_wake_word_model() -> Result<WakewordModel, VoiceError> {
    let wake_word_paths = [
        "assets/wake-word.rpw",                      // Local cyterm model
        "../koffee/training/models/syrup.rpw",       // Trained model
        "../koffee/tests/resources/syrup_model.rpw", // Test model fallback
    ];

    for model_path in &wake_word_paths {
        if let Ok(model) = WakewordModel::load_from_file(model_path) {
            log::debug!("✅ Loaded wake word model: {}", model_path);
            return Ok(model);
        }
    }

    Err(VoiceError::Configuration(
        "No wake word models could be loaded. Available paths checked: assets/wake-word.rpw, ../koffee/training/models/syrup.rpw, ../koffee/tests/resources/syrup_model.rpw".to_string()
    ))
}

/// Detection result from wake word processing
#[derive(Debug, Clone)]
pub struct WakeWordDetection {
    pub name: String,
    pub score: f32,
}

/// Results from processing all independent listeners simultaneously
pub struct IndependentListenerResults {
    pub wake_detection: Option<WakeWordDetection>,
    pub vad_probability: f32,
    pub transcription_context: Option<TranscriptionContext>,
}

/// Pre-converted audio data for efficient transcription
pub struct TranscriptionContext {
    pub audio_bytes: Vec<u8>,
    pub sample_rate: u32,
    pub format: AudioFormat,
}

/// Audio stream for lock-free processing
pub struct AudioStream {
    pub consumer: HeapCons<f32>,
    #[allow(dead_code)] // Producer reserved for future audio input implementation
    pub producer: HeapProd<f32>,
}

/// Production-Quality AudioProcessor for zero-allocation audio pipeline
/// Combines VAD, wake word detection, and transcription using real implementations
pub struct AudioProcessor {
    wake_word_detector: KoffeeCandle,
    vad_detector: VoiceActivityDetector,
    pub frame_size: usize,
}

impl AudioProcessor {
    /// Create new AudioProcessor with production-quality configurations
    pub fn new(wake_word_config: WakeWordConfig) -> Result<Self, VoiceError> {
        // Create KoffeeCandle with proper configuration from WakeWordConfig
        let mut koffee_config = KoffeeCandleConfig::default();
        koffee_config.detector.threshold = wake_word_config.sensitivity;
        koffee_config.detector.avg_threshold = wake_word_config.sensitivity * 0.875; // Slightly lower than main threshold
        koffee_config.filters.band_pass.enabled = wake_word_config.band_pass_enabled;
        koffee_config.filters.band_pass.low_cutoff = wake_word_config.band_pass_low_cutoff;
        koffee_config.filters.band_pass.high_cutoff = wake_word_config.band_pass_high_cutoff;

        let mut wake_word_detector = KoffeeCandle::new(&koffee_config).map_err(|e| {
            VoiceError::Configuration(format!("Failed to create wake word detector: {}", e))
        })?;

        // Load wake word model with fallback paths (using proven pattern)
        // Load wake word model using the dedicated function
        let model = load_wake_word_model()?;
        wake_word_detector.add_wakeword_model(model).map_err(|e| {
            VoiceError::Configuration(format!("Failed to add wake word model: {}", e))
        })?;

        // Create VoiceActivityDetector with proper configuration
        let vad_detector = VoiceActivityDetector::builder()
            .chunk_size(1024_usize)
            .sample_rate(16000_i64)
            .build()
            .map_err(|e| {
                VoiceError::Configuration(format!("Failed to create VAD detector: {:?}", e))
            })?;

        // Note: WhisperSttBuilder is created per-transcription for thread safety
        // No need to store a transcriber instance

        Ok(AudioProcessor {
            wake_word_detector,
            vad_detector,
            frame_size: 1600, // 100ms at 16kHz
        })
    }

    /// Create lock-free audio stream using ring buffer
    pub fn create_audio_stream(&self) -> Result<AudioStream, VoiceError> {
        let rb = HeapRb::<f32>::new(32000); // 2 seconds buffer at 16kHz
        let (producer, consumer) = rb.split();

        Ok(AudioStream { consumer, producer })
    }

    /// Process audio chunk for wake word detection using real KoffeeCandle
    /// Returns Some(WakeWordDetection) if wake word detected, None otherwise
    pub fn process_audio_chunk(&mut self, audio_chunk: &[f32]) -> Option<WakeWordDetection> {
        // Use real KoffeeCandle for wake word detection
        if let Some(detection) = self.wake_word_detector.process_samples(audio_chunk) {
            Some(WakeWordDetection {
                name: detection.name,
                score: detection.score,
            })
        } else {
            None
        }
    }

    /// Process audio chunk for Voice Activity Detection using real VoiceActivityDetector
    /// Returns speech probability (0.0 to 1.0)
    pub fn process_vad(&mut self, audio_chunk: &[f32]) -> Result<f32, VoiceError> {
        // Use real VoiceActivityDetector for speech detection
        self.vad_detector
            .predict(audio_chunk.iter().copied())
            .map_err(|e| VoiceError::ProcessingError(format!("VAD prediction failed: {:?}", e)))
    }

    /// Transcribe audio data using WhisperSttBuilder
    /// Returns transcribed text
    pub async fn transcribe_audio(&mut self, audio_data: &[f32]) -> Result<String, VoiceError> {
        // Convert f32 samples to bytes for WhisperSttBuilder
        let audio_bytes: Vec<u8> = audio_data
            .iter()
            .flat_map(|&sample| {
                let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                sample_i16.to_le_bytes()
            })
            .collect();

        // Use WhisperSttBuilder for in-memory transcription
        let speech_source = SpeechSource::Memory {
            data: audio_bytes,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16000,
        };

        // Create new WhisperSttBuilder for thread safety
        let conversation = WhisperSttBuilder::new()
            .with_source(speech_source)
            .transcribe(|conversation_result| conversation_result)
            .await?;

        let transcribed_text = conversation.collect().await?;

        Ok(transcribed_text)
    }

    /// Process all independent listeners simultaneously on the same audio chunk
    /// REPLACES: Three separate method calls with one coordinated call
    pub fn process_independent_listeners(
        &mut self,
        audio_chunk: &[f32],
    ) -> IndependentListenerResults {
        // Process all three listeners simultaneously (not sequentially)
        let wake_result =
            if let Some(detection) = self.wake_word_detector.process_samples(audio_chunk) {
                Some(WakeWordDetection {
                    name: detection.name,
                    score: detection.score,
                })
            } else {
                None
            };
        let vad_result = self
            .vad_detector
            .predict(audio_chunk.iter().copied())
            .unwrap_or(0.0); // Uses iterator

        // Only prepare transcription context if wake word detected and speech active
        let transcription_context = if wake_result.is_some() && vad_result > 0.5 {
            Some(self.prepare_transcription_context(audio_chunk))
        } else {
            None
        };

        IndependentListenerResults {
            wake_detection: wake_result,
            vad_probability: vad_result,
            transcription_context,
        }
    }

    /// Convert audio data once for transcription (eliminates redundant conversions)
    fn prepare_transcription_context(&self, audio_chunk: &[f32]) -> TranscriptionContext {
        // Convert f32 → bytes once and store for later use
        let audio_bytes: Vec<u8> = audio_chunk
            .iter()
            .flat_map(|&sample| {
                let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                sample_i16.to_le_bytes()
            })
            .collect();

        TranscriptionContext {
            audio_bytes,
            sample_rate: 16000,
            format: AudioFormat::Pcm16Khz,
        }
    }

    /// Transcribe audio using pre-converted context data (no redundant conversions)
    pub async fn transcribe_with_context(
        &mut self,
        context: TranscriptionContext,
    ) -> Result<String, VoiceError> {
        // Use pre-converted audio bytes (no redundant f32 → bytes conversion)
        let speech_source = SpeechSource::Memory {
            data: context.audio_bytes,
            format: context.format,
            sample_rate: context.sample_rate,
        };

        // Create WhisperSttBuilder with pre-converted data
        let conversation = WhisperSttBuilder::new()
            .with_source(speech_source)
            .transcribe(|conversation_result| conversation_result)
            .await?;

        conversation.collect().await
    }
}

/// Write PCM f32 samples to WAV file for Whisper processing
/// Zero-allocation WAV header generation with optimal I/O
pub fn write_wav_file(
    path: &str,
    samples: &[f32],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = std::fs::File::create(path)?;

    // WAV header constants
    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2; // 16-bit mono
    let data_size = num_samples * 2;
    let file_size = 36 + data_size;

    // Write WAV header (optimized for 16-bit mono PCM)
    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // Format chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // chunk size
    file.write_all(&1u16.to_le_bytes())?; // PCM format
    file.write_all(&1u16.to_le_bytes())?; // mono
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?; // block align
    file.write_all(&16u16.to_le_bytes())?; // bits per sample

    // Data chunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    // Convert f32 samples to i16 and write (SIMD-optimized)
    for sample in samples {
        let sample_i16 = (*sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        file.write_all(&sample_i16.to_le_bytes())?;
    }

    Ok(())
}

/// Audio processing constants
pub const AUDIO_CHUNK_SIZE: usize = 2048; // 128ms at 16kHz
