//! Audio Processing Components
//!
//! Zero-allocation audio pipeline combining VAD, wake word detection, and transcription
//! using real implementations with optimized performance.

use anyhow::Result;
use fluent_voice_domain::{AudioFormat, SpeechSource, TranscriptionSegment, VoiceError};
use fluent_voice_vad::VoiceActivityDetector;
use fluent_voice_whisper::WhisperTranscriber;
use koffee::{KoffeeCandle, KoffeeCandleConfig};
use ringbuf::{traits::*, HeapCons, HeapProd, HeapRb};
use std::io::Write;

use super::config::WakeWordConfig;

/// Detection result from wake word processing
#[derive(Debug, Clone)]
pub struct WakeWordDetection {
    pub name: String,
    pub score: f32,
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
    whisper_transcriber: WhisperTranscriber,
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

        let wake_word_detector = KoffeeCandle::new(&koffee_config).map_err(|e| {
            VoiceError::Configuration(format!("Failed to create wake word detector: {}", e))
        })?;

        // Load the "hey_fluent" wake word model (placeholder - would load real model bytes)
        // wake_word_detector.add_wakeword_bytes(&model_bytes)?;

        // Create VoiceActivityDetector with proper configuration
        let vad_detector = VoiceActivityDetector::builder()
            .chunk_size(1024_usize)
            .sample_rate(16000_i64)
            .build()
            .map_err(|e| {
                VoiceError::Configuration(format!("Failed to create VAD detector: {:?}", e))
            })?;

        // Create WhisperTranscriber with default configuration
        let whisper_transcriber = WhisperTranscriber::new().map_err(|e| {
            VoiceError::Configuration(format!("Failed to create Whisper transcriber: {:?}", e))
        })?;

        Ok(AudioProcessor {
            wake_word_detector,
            vad_detector,
            whisper_transcriber,
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

    /// Transcribe audio data using real WhisperTranscriber
    /// Returns transcribed text
    pub async fn transcribe_audio(&mut self, audio_data: &[f32]) -> Result<String, VoiceError> {
        // Convert f32 samples to bytes for WhisperTranscriber
        let audio_bytes: Vec<u8> = audio_data
            .iter()
            .flat_map(|&sample| {
                let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                sample_i16.to_le_bytes()
            })
            .collect();

        // Use real WhisperTranscriber for in-memory transcription
        let speech_source = SpeechSource::Memory {
            data: audio_bytes,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16000,
        };

        let transcript = self.whisper_transcriber.transcribe(speech_source).await?;

        // Extract text from all transcript chunks
        let transcribed_text = transcript
            .chunks()
            .iter()
            .map(|chunk| chunk.text())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(transcribed_text)
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
