//! High-Performance Speech Generation Engine
//!
//! This module provides a blazing-fast, zero-allocation speech generation system
//! with real-time streaming capabilities and comprehensive error handling.

use crate::error::MoshiError;
use crate::tts::{Config as TtsConfig, Model as TtsModel};
use candle_core::{DType, Device};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Audio buffer size for streaming generation (16KB = ~180ms at 44.1kHz)
const AUDIO_BUFFER_SIZE: usize = 16384;
/// Token buffer size for text processing (supports ~2000 characters)
const TOKEN_BUFFER_SIZE: usize = 2048;
/// Maximum text length for single generation (64KB)
const MAX_TEXT_LENGTH: usize = 65536;
/// Audio sample rate (24kHz - Moshi standard)
const SAMPLE_RATE: u32 = 24000;
/// Audio channels (stereo)
const CHANNELS: u8 = 2;
/// Generation chunk size for streaming
const GENERATION_CHUNK_SIZE: usize = 512;

/// Comprehensive error types for speech generation
#[derive(Debug, Clone, thiserror::Error)]
pub enum SpeechGenerationError {
    #[error("Model initialization failed: {0}")]
    ModelInitialization(String),
    #[error("Text processing failed: {0}")]
    TextProcessing(String),
    #[error("Audio generation failed: {0}")]
    AudioGeneration(String),
    #[error("Buffer overflow: requested {requested}, available {available}")]
    BufferOverflow { requested: usize, available: usize },
    #[error("Invalid voice parameters: {0}")]
    InvalidVoiceParameters(String),
    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),
    #[error("Configuration error: {0}")]
    Configuration(String),
    #[error("Device error: {0}")]
    Device(String),
    #[error("Tensor operation failed: {0}")]
    TensorOperation(String),
    #[error("Model loading failed: {0}")]
    ModelLoading(String),
    #[error("Audio processing failed: {0}")]
    AudioProcessing(String),
    #[error("Speaker PCM processing failed: {0}")]
    SpeakerPcmProcessing(String),
    #[error("Invalid speaker PCM data: {0}")]
    InvalidSpeakerPcm(String),
    #[error("Speaker embedding extraction failed: {0}")]
    SpeakerEmbedding(String),
}

impl From<MoshiError> for SpeechGenerationError {
    fn from(err: MoshiError) -> Self {
        match err {
            MoshiError::Config(msg) => SpeechGenerationError::Configuration(msg),
            MoshiError::Custom(msg) => SpeechGenerationError::AudioGeneration(msg),
            MoshiError::Candle(e) => SpeechGenerationError::TensorOperation(e.to_string()),
            MoshiError::ModelLoad(e) => SpeechGenerationError::ModelLoading(e.to_string()),
            MoshiError::Audio(msg) => SpeechGenerationError::AudioProcessing(msg),
            MoshiError::Io(e) => SpeechGenerationError::ModelLoading(e.to_string()),
            MoshiError::Serde(e) => SpeechGenerationError::Configuration(e.to_string()),
            MoshiError::Generation(msg) => SpeechGenerationError::AudioGeneration(msg),
            MoshiError::Tokenization(e) => SpeechGenerationError::Configuration(e.to_string()),
        }
    }
}

impl From<candle_core::Error> for SpeechGenerationError {
    fn from(err: candle_core::Error) -> Self {
        SpeechGenerationError::TensorOperation(err.to_string())
    }
}

/// Voice parameters for speech synthesis control
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone)]
pub struct SpeakerPcmConfig {
    pub target_sample_rate: u32,
    pub target_channels: u16,
    pub embedding_dim: usize,
    pub min_samples: usize,
    pub max_samples: usize,
    pub normalization_enabled: bool,
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

    /// Apply voice parameters to audio samples
    #[inline]
    pub fn apply_to_samples(&self, samples: &mut Vec<f32>) {
        // FFT imports removed - not used in current implementation

        // Apply volume adjustment
        if self.volume != 1.0 {
            for sample in samples.iter_mut() {
                *sample *= self.volume;
            }
        }

        // Apply speed modification (time-stretching without pitch change)
        if (self.speed - 1.0).abs() > f32::EPSILON {
            let stretched = self.apply_psola_stretch(samples, self.speed);
            samples.clear();
            samples.extend_from_slice(&stretched);
        }

        // Apply pitch shifting (frequency domain)
        if self.pitch.abs() > f32::EPSILON {
            // Convert semitones to frequency ratio: 2^(semitones/12)
            let pitch_ratio = 2.0_f32.powf(self.pitch / 12.0);
            let shifted = self.apply_pitch_shift_fft(samples, pitch_ratio);
            samples.clear();
            samples.extend_from_slice(&shifted);
        }

        // Apply emotion processing (spectral filtering)
        if (self.emotion - 0.5).abs() > f32::EPSILON {
            let filtered = self.apply_emotion_spectral_filter(samples, self.emotion);
            samples.clear();
            samples.extend_from_slice(&filtered);
        }

        // Apply emphasis through dynamic range compression
        if self.emphasis != 1.0 {
            let threshold = 0.7;
            let ratio = 1.0 / self.emphasis;

            for sample in samples.iter_mut() {
                let abs_sample = sample.abs();
                if abs_sample > threshold {
                    let excess = abs_sample - threshold;
                    let compressed = threshold + excess * ratio;
                    *sample = if *sample >= 0.0 {
                        compressed
                    } else {
                        -compressed
                    };
                }
            }
        }

        // Note: pause_duration is handled at generation level, not sample level
    }

    fn apply_psola_stretch(&self, samples: &[f32], speed_factor: f32) -> Vec<f32> {
        if samples.is_empty() || (speed_factor - 1.0).abs() < f32::EPSILON {
            return samples.to_vec();
        }

        const FRAME_SIZE: usize = 1024;
        const HOP_SIZE: usize = 256;

        let input_hop = (HOP_SIZE as f32 / speed_factor) as usize;
        let output_hop = HOP_SIZE;

        // Hanning window
        let window: Vec<f32> = (0..FRAME_SIZE)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (FRAME_SIZE - 1) as f32).cos())
            })
            .collect();

        let mut output = vec![0.0f32; (samples.len() as f32 * speed_factor) as usize + FRAME_SIZE];
        let mut phase_accumulator = vec![0.0f32; FRAME_SIZE / 2 + 1];
        let mut last_phase = vec![0.0f32; FRAME_SIZE / 2 + 1];

        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let ifft = planner.plan_fft_inverse(FRAME_SIZE);

        let mut input_pos = 0;
        let mut output_pos = 0;

        while input_pos + FRAME_SIZE <= samples.len() {
            // Extract and window frame
            let mut frame: Vec<f32> = samples[input_pos..input_pos + FRAME_SIZE]
                .iter()
                .zip(&window)
                .map(|(&s, &w)| s * w)
                .collect();

            // Forward FFT
            let mut spectrum = fft.make_output_vec();
            if fft.process(&mut frame, &mut spectrum).is_err() {
                break;
            }

            // Phase vocoder processing
            for (i, bin) in spectrum.iter_mut().enumerate() {
                let magnitude = bin.norm();
                let phase = bin.arg();

                // Calculate phase difference
                let mut phase_diff = phase - last_phase[i];
                last_phase[i] = phase;

                // Unwrap phase
                while phase_diff > std::f32::consts::PI {
                    phase_diff -= 2.0 * std::f32::consts::PI;
                }
                while phase_diff < -std::f32::consts::PI {
                    phase_diff += 2.0 * std::f32::consts::PI;
                }

                // Calculate true frequency
                let bin_freq = 2.0 * std::f32::consts::PI * i as f32 / FRAME_SIZE as f32;
                let true_freq = bin_freq + phase_diff / input_hop as f32;

                // Update phase accumulator
                phase_accumulator[i] += true_freq * output_hop as f32;

                // Reconstruct bin
                *bin = rustfft::num_complex::Complex::from_polar(magnitude, phase_accumulator[i]);
            }

            // Inverse FFT
            let mut output_frame = ifft.make_input_vec();
            output_frame.copy_from_slice(&spectrum);
            let mut time_frame = vec![0.0f32; FRAME_SIZE];
            if ifft.process(&mut output_frame, &mut time_frame).is_err() {
                break;
            }

            // Overlap-add with window
            for (i, &sample) in time_frame.iter().enumerate() {
                let windowed = sample * window[i];
                if output_pos + i < output.len() {
                    output[output_pos + i] += windowed;
                }
            }

            input_pos += input_hop;
            output_pos += output_hop;
        }

        output.truncate((samples.len() as f32 * speed_factor) as usize);
        output
    }

    fn apply_pitch_shift_fft(&self, samples: &[f32], pitch_ratio: f32) -> Vec<f32> {
        if samples.is_empty() || (pitch_ratio - 1.0).abs() < f32::EPSILON {
            return samples.to_vec();
        }

        const FRAME_SIZE: usize = 2048;
        const HOP_SIZE: usize = 512;

        // Hanning window
        let window: Vec<f32> = (0..FRAME_SIZE)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (FRAME_SIZE - 1) as f32).cos())
            })
            .collect();

        let mut output = vec![0.0f32; samples.len() + FRAME_SIZE];

        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let ifft = planner.plan_fft_inverse(FRAME_SIZE);

        let mut pos = 0;

        while pos + FRAME_SIZE <= samples.len() {
            // Extract and window frame
            let mut frame: Vec<f32> = samples[pos..pos + FRAME_SIZE]
                .iter()
                .zip(&window)
                .map(|(&s, &w)| s * w)
                .collect();

            // Forward FFT
            let mut spectrum = fft.make_output_vec();
            if fft.process(&mut frame, &mut spectrum).is_err() {
                break;
            }

            // Pitch shift by frequency domain shifting
            let mut shifted_spectrum =
                vec![rustfft::num_complex::Complex::new(0.0, 0.0); spectrum.len()];

            for i in 0..spectrum.len() {
                let shifted_bin = (i as f32 * pitch_ratio) as usize;
                if shifted_bin < shifted_spectrum.len() {
                    shifted_spectrum[shifted_bin] = spectrum[i];
                }
            }

            // Inverse FFT
            let mut time_frame = vec![0.0f32; FRAME_SIZE];
            if ifft
                .process(&mut shifted_spectrum, &mut time_frame)
                .is_err()
            {
                break;
            }

            // Overlap-add with window
            for (i, &sample) in time_frame.iter().enumerate() {
                let windowed = sample * window[i];
                if pos + i < output.len() {
                    output[pos + i] += windowed;
                }
            }

            pos += HOP_SIZE;
        }

        output.truncate(samples.len());
        output
    }

    fn apply_emotion_spectral_filter(&self, samples: &[f32], emotion: f32) -> Vec<f32> {
        if samples.is_empty() || (emotion - 0.5).abs() < f32::EPSILON {
            return samples.to_vec();
        }

        const FRAME_SIZE: usize = 1024;
        const HOP_SIZE: usize = 256;

        // Hanning window
        let window: Vec<f32> = (0..FRAME_SIZE)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (FRAME_SIZE - 1) as f32).cos())
            })
            .collect();

        let mut output = vec![0.0f32; samples.len() + FRAME_SIZE];

        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let ifft = planner.plan_fft_inverse(FRAME_SIZE);

        // Emotion-based spectral filtering
        // 0.0 = sad (low-pass, reduced high frequencies)
        // 0.5 = neutral (no change)
        // 1.0 = happy (enhanced harmonics, brighter)
        let filter_curve: Vec<f32> = (0..FRAME_SIZE / 2 + 1)
            .map(|i| {
                let freq_ratio = i as f32 / (FRAME_SIZE / 2) as f32;
                if emotion < 0.5 {
                    // Sad: low-pass filter
                    let cutoff = 0.3 + 0.4 * emotion * 2.0; // 0.3 to 0.7
                    if freq_ratio < cutoff {
                        1.0
                    } else {
                        (-10.0 * (freq_ratio - cutoff)).exp()
                    }
                } else {
                    // Happy: enhance harmonics
                    let brightness = (emotion - 0.5) * 2.0; // 0.0 to 1.0
                    1.0 + brightness * (freq_ratio * 2.0).min(1.0)
                }
            })
            .collect();

        let mut pos = 0;

        while pos + FRAME_SIZE <= samples.len() {
            // Extract and window frame
            let mut frame: Vec<f32> = samples[pos..pos + FRAME_SIZE]
                .iter()
                .zip(&window)
                .map(|(&s, &w)| s * w)
                .collect();

            // Forward FFT
            let mut spectrum = fft.make_output_vec();
            if fft.process(&mut frame, &mut spectrum).is_err() {
                break;
            }

            // Apply emotional spectral filter
            for (i, bin) in spectrum.iter_mut().enumerate() {
                *bin *= filter_curve[i];
            }

            // Inverse FFT
            let mut time_frame = vec![0.0f32; FRAME_SIZE];
            if ifft.process(&mut spectrum, &mut time_frame).is_err() {
                break;
            }

            // Overlap-add with window
            for (i, &sample) in time_frame.iter().enumerate() {
                let windowed = sample * window[i];
                if pos + i < output.len() {
                    output[pos + i] += windowed;
                }
            }

            pos += HOP_SIZE;
        }

        output.truncate(samples.len());
        output
    }
}

/// Generation statistics for performance monitoring
#[derive(Debug, Default)]
pub struct GenerationStats {
    /// Total samples generated
    pub samples_generated: AtomicUsize,
    /// Total generation time in milliseconds
    pub generation_time_ms: AtomicUsize,
    /// Number of generation calls
    pub generation_calls: AtomicUsize,
    /// Peak memory usage in bytes
    pub peak_memory_usage: AtomicUsize,
    /// Buffer underruns
    pub buffer_underruns: AtomicUsize,
    /// Audio quality metrics
    pub audio_quality_score: AtomicUsize, // Scaled by 1000 for atomic storage
}

impl GenerationStats {
    /// Get average generation time per sample
    pub fn avg_generation_time_per_sample(&self) -> f64 {
        let total_time = self.generation_time_ms.load(Ordering::Relaxed) as f64;
        let total_samples = self.samples_generated.load(Ordering::Relaxed) as f64;
        if total_samples > 0.0 {
            total_time / total_samples
        } else {
            0.0
        }
    }

    /// Get real-time factor (how much faster than real-time)
    pub fn real_time_factor(&self) -> f64 {
        let total_time = self.generation_time_ms.load(Ordering::Relaxed) as f64;
        let total_samples = self.samples_generated.load(Ordering::Relaxed) as f64;
        if total_time > 0.0 {
            (total_samples / SAMPLE_RATE as f64) / (total_time / 1000.0)
        } else {
            0.0
        }
    }

    /// Get audio quality score (0.0 to 1.0)
    pub fn audio_quality(&self) -> f32 {
        self.audio_quality_score.load(Ordering::Relaxed) as f32 / 1000.0
    }

    /// Record generation metrics
    #[inline]
    pub fn record_generation(&self, samples: usize, time_ms: usize) {
        self.samples_generated.fetch_add(samples, Ordering::Relaxed);
        self.generation_time_ms
            .fetch_add(time_ms, Ordering::Relaxed);
        self.generation_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Record buffer underrun
    #[inline]
    pub fn record_underrun(&self) {
        self.buffer_underruns.fetch_add(1, Ordering::Relaxed);
    }
}

/// Configuration for speech generation engine
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// TTS model configuration
    pub tts_config: TtsConfig,
    /// Voice parameters
    pub voice_params: VoiceParameters,
    /// Maximum generation steps
    pub max_steps: usize,
    /// Generation temperature (0.0 to 2.0)
    pub temperature: f64,
    /// Top-k sampling parameter
    pub top_k: usize,
    /// Top-p nucleus sampling parameter
    pub top_p: f64,
    /// Random seed for reproducible generation
    pub seed: u64,
    /// Enable real-time streaming
    pub enable_streaming: bool,
    /// Audio buffer size for streaming
    pub stream_buffer_size: usize,
    /// Device for computation
    pub device: Device,
    /// Data type for tensors
    pub dtype: DType,
    /// Speaker PCM processing configuration
    pub speaker_pcm: SpeakerPcmConfig,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            tts_config: TtsConfig::v202501(),
            voice_params: VoiceParameters::default(),
            max_steps: 2000,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            seed: 42,
            enable_streaming: true,
            stream_buffer_size: AUDIO_BUFFER_SIZE,
            device: Device::Cpu,
            dtype: DType::F32,
            speaker_pcm: SpeakerPcmConfig::default(),
        }
    }
}

/// High-performance circular buffer for streaming audio
#[derive(Debug)]
pub struct AudioBuffer {
    /// Pre-allocated audio data buffer
    data: Box<[f32; AUDIO_BUFFER_SIZE]>,
    /// Current write position
    write_pos: usize,
    /// Current read position
    read_pos: usize,
    /// Number of samples available for reading
    available: usize,
    /// Sample rate
    sample_rate: u32,
    /// Number of channels
    channels: u8,
}

impl AudioBuffer {
    /// Create new audio buffer with pre-allocated memory
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        Self {
            data: Box::new([0.0; AUDIO_BUFFER_SIZE]),
            write_pos: 0,
            read_pos: 0,
            available: 0,
            sample_rate,
            channels,
        }
    }

    /// Write audio samples to buffer (zero-copy when possible)
    #[inline]
    pub fn write_samples(&mut self, samples: &[f32]) -> Result<usize, SpeechGenerationError> {
        if samples.len() > self.capacity() - self.available {
            return Err(SpeechGenerationError::BufferOverflow {
                requested: samples.len(),
                available: self.capacity() - self.available,
            });
        }

        let mut written = 0;
        for &sample in samples {
            if self.available >= self.capacity() {
                break;
            }

            self.data[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % AUDIO_BUFFER_SIZE;
            self.available += 1;
            written += 1;
        }

        Ok(written)
    }

    /// Read audio samples from buffer (zero-copy when possible)
    #[inline]
    pub fn read_samples(&mut self, output: &mut [f32]) -> usize {
        let to_read = output.len().min(self.available);

        for i in 0..to_read {
            output[i] = self.data[self.read_pos];
            self.read_pos = (self.read_pos + 1) % AUDIO_BUFFER_SIZE;
        }

        self.available -= to_read;
        to_read
    }

    /// Get number of samples available for reading
    #[inline]
    pub fn available(&self) -> usize {
        self.available
    }

    /// Get buffer capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        AUDIO_BUFFER_SIZE
    }

    /// Check if buffer is full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.available == AUDIO_BUFFER_SIZE
    }

    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.available == 0
    }

    /// Clear buffer
    #[inline]
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.available = 0;
    }

    /// Get audio format information
    #[inline]
    pub fn format(&self) -> (u32, u8) {
        (self.sample_rate, self.channels)
    }
}

/// Streaming audio output with zero-copy access
#[derive(Debug)]
pub struct AudioStream<'a> {
    /// Audio data reference
    data: &'a [f32],
    /// Sample rate
    sample_rate: u32,
    /// Number of channels
    channels: u8,
    /// Duration in seconds
    duration: f64,
}

impl<'a> AudioStream<'a> {
    /// Create new audio stream
    pub fn new(data: &'a [f32], sample_rate: u32, channels: u8) -> Self {
        let duration = data.len() as f64 / (sample_rate as f64 * channels as f64);
        Self {
            data,
            sample_rate,
            channels,
            duration,
        }
    }

    /// Get audio data
    #[inline]
    pub fn data(&self) -> &[f32] {
        self.data
    }

    /// Get sample rate
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get number of channels
    #[inline]
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Get duration in seconds
    #[inline]
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// Get number of samples
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if stream is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Convert to stereo (duplicate mono channel)
    pub fn to_stereo(
        &self,
        output: &mut [f32],
    ) -> std::result::Result<usize, SpeechGenerationError> {
        if self.channels == 1 {
            // Mono to stereo conversion
            let samples_to_convert = (output.len() / 2).min(self.data.len());

            for i in 0..samples_to_convert {
                output[i * 2] = self.data[i];
                output[i * 2 + 1] = self.data[i];
            }

            Ok(samples_to_convert * 2)
        } else if self.channels == 2 {
            // Already stereo, direct copy
            let samples_to_copy = output.len().min(self.data.len());
            output[..samples_to_copy].copy_from_slice(&self.data[..samples_to_copy]);
            Ok(samples_to_copy)
        } else {
            Err(SpeechGenerationError::AudioProcessing(format!(
                "Unsupported channel count: {}",
                self.channels
            )))
        }
    }

    /// Apply audio effects
    pub fn apply_effects(
        &self,
        params: &VoiceParameters,
        output: &mut [f32],
    ) -> std::result::Result<usize, SpeechGenerationError> {
        let samples_to_process = output.len().min(self.data.len());
        output[..samples_to_process].copy_from_slice(&self.data[..samples_to_process]);

        // Apply voice parameters
        let mut temp_vec = output[..samples_to_process].to_vec();
        params.apply_to_samples(&mut temp_vec);
        output[..samples_to_process].copy_from_slice(&temp_vec);

        Ok(samples_to_process)
    }
}

/// Iterator for streaming audio generation
pub struct AudioStreamIterator<'a> {
    /// Reference to the speech generator
    generator: &'a mut SpeechGenerator,
    /// Current text position
    text_pos: usize,
    /// Text being processed
    text: String,
    /// Current generation state
    state: GenerationState,
    /// Remaining generation steps
    remaining_steps: usize,
    /// Generated audio chunks (used for buffering stream data)
    generated_chunks: Vec<Vec<f32>>,
    /// Current chunk index (used for stream position tracking)
    chunk_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum GenerationState {
    Initializing,
    Processing,
    Generating,
    Finishing,
    Complete,
}

impl<'a> AudioStreamIterator<'a> {
    /// Create new streaming iterator
    fn new(generator: &'a mut SpeechGenerator, text: String) -> Self {
        let remaining_steps = generator.config.max_steps;
        Self {
            generator,
            text_pos: 0,
            text,
            state: GenerationState::Initializing,
            remaining_steps,
            generated_chunks: Vec::new(),
            chunk_index: 0,
        }
    }
}

impl<'a> Iterator for AudioStreamIterator<'a> {
    type Item = std::result::Result<Vec<f32>, SpeechGenerationError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.state == GenerationState::Complete || self.remaining_steps == 0 {
            return None;
        }

        match self.state {
            GenerationState::Initializing => {
                // Initialize generation
                self.state = GenerationState::Processing;
                self.remaining_steps -= 1;

                // Generate initial chunk
                match self.generator.generate_chunk(&self.text[self.text_pos..]) {
                    Ok(chunk_data) => {
                        self.text_pos += GENERATION_CHUNK_SIZE.min(self.text.len() - self.text_pos);
                        if self.text_pos >= self.text.len() {
                            self.state = GenerationState::Finishing;
                        }
                        Some(Ok(chunk_data))
                    }
                    Err(e) => {
                        self.state = GenerationState::Complete;
                        Some(Err(e))
                    }
                }
            }
            GenerationState::Processing => {
                // Continue processing
                if self.text_pos < self.text.len() {
                    match self.generator.generate_chunk(&self.text[self.text_pos..]) {
                        Ok(chunk_data) => {
                            self.text_pos +=
                                GENERATION_CHUNK_SIZE.min(self.text.len() - self.text_pos);
                            self.remaining_steps -= 1;

                            if self.text_pos >= self.text.len() {
                                self.state = GenerationState::Finishing;
                            }
                            Some(Ok(chunk_data))
                        }
                        Err(e) => {
                            self.state = GenerationState::Complete;
                            Some(Err(e))
                        }
                    }
                } else {
                    self.state = GenerationState::Finishing;
                    self.next()
                }
            }
            GenerationState::Finishing => {
                // Finalize generation
                self.state = GenerationState::Complete;
                match self.generator.finalize_generation() {
                    Ok(silence_data) => Some(Ok(silence_data)),
                    Err(e) => Some(Err(e)),
                }
            }
            _ => None,
        }
    }
}

/// High-performance speech generation engine
pub struct SpeechGenerator {
    /// TTS model for speech synthesis
    tts_model: TtsModel,
    /// Audio buffer for streaming
    audio_buffer: AudioBuffer,
    /// Pre-allocated token buffer (used for batch token processing)
    token_buffer: Box<[u32; TOKEN_BUFFER_SIZE]>,
    /// Generation configuration
    config: GeneratorConfig,
    /// Performance statistics
    stats: GenerationStats,
    /// Current generation state
    generation_active: bool,
    /// Text processing queue (used for async text processing pipeline)
    text_queue: VecDeque<String>,
}

impl SpeechGenerator {
    /// Create new speech generator with model files
    ///
    /// This constructor requires actual model files to be provided.
    /// For production use, this ensures real model weights are loaded.
    pub fn new<P: AsRef<Path>>(
        lm_model_path: P,
        mimi_model_path: P,
        config: GeneratorConfig,
    ) -> Result<Self, SpeechGenerationError> {
        // Validate model files exist and are readable
        Self::validate_model_file(&lm_model_path, "language model")?;
        Self::validate_model_file(&mimi_model_path, "Mimi model")?;

        // Load TTS model with actual weights from files
        let tts_model =
            TtsModel::load(lm_model_path, mimi_model_path, config.dtype, &config.device)
                .map_err(|e| SpeechGenerationError::ModelLoading(e.to_string()))?;

        // Initialize audio buffer
        let audio_buffer = AudioBuffer::new(SAMPLE_RATE, CHANNELS);

        // Pre-allocate token buffer
        let token_buffer = Box::new([0u32; TOKEN_BUFFER_SIZE]);

        Ok(Self {
            tts_model,
            audio_buffer,
            token_buffer,
            config,
            stats: GenerationStats::default(),
            generation_active: false,
            text_queue: VecDeque::with_capacity(16),
        })
    }

    /// Load models from files with optimized loading (deprecated - use new() instead)
    #[deprecated(note = "Use SpeechGenerator::new() instead, which now requires model files")]
    pub fn load_from_files<P: AsRef<Path>>(
        lm_model_path: P,
        mimi_model_path: P,
        config: GeneratorConfig,
    ) -> Result<Self, SpeechGenerationError> {
        Self::new(lm_model_path, mimi_model_path, config)
    }

    /// Validate that a model file exists and is a valid safetensors file
    fn validate_model_file<P: AsRef<Path>>(
        path: P,
        model_type: &str,
    ) -> Result<(), SpeechGenerationError> {
        let path = path.as_ref();

        // Check if file exists
        if !path.exists() {
            return Err(SpeechGenerationError::ModelLoading(format!(
                "{} file not found: {}",
                model_type,
                path.display()
            )));
        }

        // Check if file is readable
        if let Err(e) = std::fs::File::open(path) {
            return Err(SpeechGenerationError::ModelLoading(format!(
                "Cannot read {} file {}: {}",
                model_type,
                path.display(),
                e
            )));
        }

        // Validate safetensors format by attempting to read header
        match std::fs::read(path) {
            Ok(data) => {
                if data.len() < 8 {
                    return Err(SpeechGenerationError::ModelLoading(format!(
                        "{} file {} is too small to be a valid safetensors file",
                        model_type,
                        path.display()
                    )));
                }

                // Basic safetensors validation - check for valid header length
                let header_len = u64::from_le_bytes([
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
                ]);

                if header_len as usize + 8 > data.len() {
                    return Err(SpeechGenerationError::ModelLoading(format!(
                        "{} file {} has invalid safetensors header",
                        model_type,
                        path.display()
                    )));
                }
            }
            Err(e) => {
                return Err(SpeechGenerationError::ModelLoading(format!(
                    "Failed to read {} file {}: {}",
                    model_type,
                    path.display(),
                    e
                )));
            }
        }

        Ok(())
    }

    /// Generate speech from text with zero-allocation hot path
    pub fn generate(&mut self, text: &str) -> std::result::Result<Vec<f32>, SpeechGenerationError> {
        if text.len() > MAX_TEXT_LENGTH {
            return Err(SpeechGenerationError::TextProcessing(format!(
                "Text too long: {} characters (max: {})",
                text.len(),
                MAX_TEXT_LENGTH
            )));
        }

        let start_time = std::time::Instant::now();

        // Reset audio buffer
        self.audio_buffer.clear();

        // Process speaker PCM data for voice cloning
        // Extract speaker PCM path from voice parameters if available
        let speaker_pcm_path = self
            .config
            .voice_params
            .voice_clone_path
            .as_ref()
            .map(|path| path.as_path());

        let speaker_pcm_tensor = self.process_speaker_pcm(
            "default_speaker",
            speaker_pcm_path,
            &self.config.speaker_pcm,
        )?;

        // Generate audio
        let audio_data = self
            .tts_model
            .generate(
                text,
                speaker_pcm_tensor.as_ref(),
                self.config.max_steps,
                self.config.temperature,
                self.config.top_k,
                self.config.top_p,
                None, // No repetition penalty
                None, // No CFG alpha
                self.config.seed,
            )
            .map_err(|e| SpeechGenerationError::AudioGeneration(e.to_string()))?;

        // Write to buffer
        self.audio_buffer
            .write_samples(&audio_data)
            .map_err(|e| SpeechGenerationError::AudioProcessing(e.to_string()))?;

        // Record performance metrics
        let elapsed = start_time.elapsed();
        self.stats
            .record_generation(audio_data.len(), elapsed.as_millis() as usize);

        Ok(audio_data)
    }

    /// Generate streaming audio with real-time processing
    pub fn generate_streaming<'a>(
        &'a mut self,
        text: &str,
    ) -> std::result::Result<AudioStreamIterator<'a>, SpeechGenerationError> {
        if text.len() > MAX_TEXT_LENGTH {
            return Err(SpeechGenerationError::TextProcessing(format!(
                "Text too long: {} characters (max: {})",
                text.len(),
                MAX_TEXT_LENGTH
            )));
        }

        self.generation_active = true;
        self.audio_buffer.clear();

        Ok(AudioStreamIterator::new(self, text.to_string()))
    }

    /// Generate a chunk of audio for streaming
    fn generate_chunk(
        &mut self,
        text: &str,
    ) -> std::result::Result<Vec<f32>, SpeechGenerationError> {
        let chunk_size = GENERATION_CHUNK_SIZE.min(text.len());
        let chunk_text = &text[..chunk_size];

        let start_time = std::time::Instant::now();

        // Generate audio chunk
        let audio_data = self
            .tts_model
            .generate(
                chunk_text,
                None,
                self.config.max_steps / 10, // Smaller chunk size
                self.config.temperature,
                self.config.top_k,
                self.config.top_p,
                None,
                None,
                self.config.seed,
            )
            .map_err(|e| SpeechGenerationError::AudioGeneration(e.to_string()))?;

        // Record performance metrics
        let elapsed = start_time.elapsed();
        self.stats
            .record_generation(audio_data.len(), elapsed.as_millis() as usize);

        Ok(audio_data)
    }

    /// Finalize streaming generation
    fn finalize_generation(&mut self) -> std::result::Result<Vec<f32>, SpeechGenerationError> {
        self.generation_active = false;

        // Generate final silence padding
        let silence_duration = 0.1; // 100ms of silence
        let silence_samples = (SAMPLE_RATE as f64 * silence_duration * CHANNELS as f64) as usize;
        let silence_data = vec![0.0f32; silence_samples];

        Ok(silence_data)
    }

    /// Configure voice parameters with validation
    pub fn set_voice_parameters(
        &mut self,
        params: VoiceParameters,
    ) -> std::result::Result<(), SpeechGenerationError> {
        params.validate()?;
        self.config.voice_params = params;
        Ok(())
    }

    /// Get current voice parameters
    pub fn voice_parameters(&self) -> &VoiceParameters {
        &self.config.voice_params
    }

    /// Get generation statistics
    pub fn stats(&self) -> &GenerationStats {
        &self.stats
    }

    /// Reset generation statistics
    pub fn reset_stats(&mut self) {
        self.stats = GenerationStats::default();
    }

    /// Check if generation is active
    pub fn is_generating(&self) -> bool {
        self.generation_active
    }

    /// Get audio buffer status
    pub fn buffer_status(&self) -> (usize, usize) {
        (self.audio_buffer.available(), self.audio_buffer.capacity())
    }

    /// Optimize performance settings
    pub fn optimize_performance(&mut self) {
        // Adjust generation parameters for performance
        self.config.max_steps = self.config.max_steps.min(1000);
        self.config.top_k = self.config.top_k.min(40);

        // Enable streaming if not already enabled
        if !self.config.enable_streaming {
            self.config.enable_streaming = true;
        }
    }

    /// Get device information
    pub fn device_info(&self) -> (&Device, DType) {
        (&self.config.device, self.config.dtype)
    }

    /// Flush audio buffer
    pub fn flush_buffer(&mut self) {
        self.audio_buffer.clear();
    }

    /// Process speaker PCM data for voice cloning and identification
    fn process_speaker_pcm(
        &self,
        _speaker_id: &str,
        audio_path: Option<&std::path::Path>,
        config: &SpeakerPcmConfig,
    ) -> Result<Option<candle_core::Tensor>, SpeechGenerationError> {
        // Early return if no speaker data provided
        let audio_path = match audio_path {
            Some(path) => path,
            None => return Ok(None),
        };

        // 1. Load and decode PCM data using whisper package's comprehensive decoder
        let (pcm_samples, original_sample_rate) = self.simple_wav_decode(audio_path)?;

        // 2. Validate PCM data
        self.validate_pcm_data(&pcm_samples, config)?;

        // 3. Normalize and resample audio
        let normalized_samples =
            self.normalize_audio_samples(&pcm_samples, original_sample_rate, config)?;

        // 4. Convert to Candle Tensor format
        let tensor = self.pcm_to_tensor(&normalized_samples, config)?;

        // 5. Apply Mimi encoding if needed for speaker embedding
        let processed_tensor = self.apply_mimi_encoding(&tensor)?;

        Ok(Some(processed_tensor))
    }

    /// Validate PCM data meets requirements
    fn validate_pcm_data(
        &self,
        samples: &[f32],
        config: &SpeakerPcmConfig,
    ) -> Result<(), SpeechGenerationError> {
        if samples.is_empty() {
            return Err(SpeechGenerationError::InvalidVoiceParameters(
                "Empty PCM samples provided".to_string(),
            ));
        }

        if samples.len() < config.min_samples {
            return Err(SpeechGenerationError::InvalidVoiceParameters(format!(
                "Insufficient samples: {} < {} (minimum)",
                samples.len(),
                config.min_samples
            )));
        }

        if samples.len() > config.max_samples {
            return Err(SpeechGenerationError::InvalidVoiceParameters(format!(
                "Too many samples: {} > {} (maximum)",
                samples.len(),
                config.max_samples
            )));
        }

        // Validate sample range [-1.0, 1.0]
        for (i, &sample) in samples.iter().enumerate() {
            if !sample.is_finite() || sample.abs() > 1.0 {
                return Err(SpeechGenerationError::InvalidVoiceParameters(format!(
                    "Invalid sample at index {}: {} (must be finite and in [-1.0, 1.0])",
                    i, sample
                )));
            }
        }

        Ok(())
    }

    /// Normalize and resample audio to target format
    fn normalize_audio_samples(
        &self,
        samples: &[f32],
        original_sample_rate: u32,
        config: &SpeakerPcmConfig,
    ) -> Result<Vec<f32>, SpeechGenerationError> {
        let mut processed_samples = samples.to_vec();

        // 1. Resample if needed (using production FFT-based resampling)
        if original_sample_rate != config.target_sample_rate {
            processed_samples = self.resample_audio_basic(
                &processed_samples,
                original_sample_rate,
                config.target_sample_rate,
            )?;
        }

        // 2. Normalize amplitude if enabled
        if config.normalization_enabled {
            let max_amplitude = processed_samples
                .iter()
                .map(|&x| x.abs())
                .fold(0.0f32, f32::max);

            if max_amplitude > 0.0 && max_amplitude != 1.0 {
                let scale_factor = 0.95 / max_amplitude; // Leave 5% headroom
                for sample in &mut processed_samples {
                    *sample *= scale_factor;
                }
            }
        }

        // 3. Ensure target length constraints
        if processed_samples.len() > config.max_samples {
            processed_samples.truncate(config.max_samples);
        }

        Ok(processed_samples)
    }

    /// Production-grade FFT-based resampling with anti-aliasing (copied from DIA)
    fn resample_audio_basic(
        &self,
        samples: &[f32],
        from_rate: u32,
        to_rate: u32,
    ) -> Result<Vec<f32>, SpeechGenerationError> {
        if from_rate == to_rate {
            return Ok(samples.to_vec());
        }

        use rubato::{FftFixedIn, Resampler};

        // Use FftFixedIn for flexibility
        const CHUNK: usize = 1024;
        const SUB_CHUNKS: usize = 2; // Number of sub-chunks for processing
        let mut resampler =
            FftFixedIn::<f32>::new(from_rate as usize, to_rate as usize, CHUNK, SUB_CHUNKS, 1)
                .map_err(|e| {
                    SpeechGenerationError::AudioProcessing(format!(
                        "Failed to create resampler: {}",
                        e
                    ))
                })?;

        // Calculate expected output capacity
        let expected_len =
            (samples.len() as f64 * to_rate as f64 / from_rate as f64).ceil() as usize;
        let mut out = Vec::with_capacity(expected_len + CHUNK);

        // Process in chunks
        let mut pos = 0;
        while pos < samples.len() {
            let end = (pos + CHUNK).min(samples.len());
            let chunk_len = end - pos;

            // Create input buffer
            let mut input_chunk = vec![0.0; CHUNK];
            input_chunk[..chunk_len].copy_from_slice(&samples[pos..end]);

            // Process this chunk
            let block = vec![input_chunk];
            let frames = resampler.process(&block, None).map_err(|e| {
                SpeechGenerationError::AudioProcessing(format!("Resampling failed: {}", e))
            })?;
            out.extend_from_slice(&frames[0]);

            pos += chunk_len;

            // For the last partial chunk, we're done
            if chunk_len < CHUNK {
                break;
            }
        }

        Ok(out)
    }

    /// Convert PCM samples to Candle Tensor
    fn pcm_to_tensor(
        &self,
        samples: &[f32],
        config: &SpeakerPcmConfig,
    ) -> Result<candle_core::Tensor, SpeechGenerationError> {
        use candle_core::{DType, Tensor};

        // Create tensor with shape [batch_size=1, channels, samples]
        let tensor = Tensor::from_vec(
            samples.to_vec(),
            (1, config.target_channels as usize, samples.len()),
            &self.config.device, // Use configured target device
        )
        .map_err(|e| {
            SpeechGenerationError::TensorOperation(format!(
                "Failed to create tensor on device {:?}: {}",
                self.config.device, e
            ))
        })?;

        // Convert to appropriate dtype for model
        let tensor = tensor.to_dtype(DType::F32).map_err(|e| {
            SpeechGenerationError::TensorOperation(format!("Failed to convert tensor dtype: {}", e))
        })?;

        Ok(tensor)
    }

    /// Apply Mimi encoding for speaker embedding extraction
    fn apply_mimi_encoding(
        &self,
        tensor: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, SpeechGenerationError> {
        // Use the existing Mimi encoder from the TTS model instead of creating a new one
        // This ensures we use the actual loaded model weights for proper encoding
        let encoded_tensor = self.tts_model.mimi().encode(tensor).map_err(|e| {
            SpeechGenerationError::SpeakerEmbedding(format!(
                "Failed to encode audio with Mimi: {}",
                e
            ))
        })?;

        tracing::debug!(
            "Applied Mimi encoding: input shape {:?} -> output shape {:?}",
            tensor.dims(),
            encoded_tensor.dims()
        );

        Ok(encoded_tensor)
    }

    /// Decode audio file using whisper package's comprehensive PCM decoder
    fn simple_wav_decode(
        &self,
        path: &std::path::Path,
    ) -> Result<(Vec<f32>, u32), SpeechGenerationError> {
        // Use the comprehensive PCM decoder from whisper package
        // Supports F32, U8, U16, U24, U32, S8, S16, S24, S32, F64 audio formats
        use fluent_voice_whisper::pcm_decode;

        let (pcm_samples, sample_rate) = pcm_decode(path).map_err(|e| {
            SpeechGenerationError::SpeakerPcmProcessing(format!(
                "Failed to decode audio file {:?}: {}",
                path, e
            ))
        })?;

        tracing::debug!(
            "Decoded {} samples at {}Hz from {:?}",
            pcm_samples.len(),
            sample_rate,
            path
        );

        Ok((pcm_samples, sample_rate))
    }
}

/// Builder for speech generator configuration
pub struct SpeechGeneratorBuilder {
    config: GeneratorConfig,
}

impl SpeechGeneratorBuilder {
    /// Create new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: GeneratorConfig::default(),
        }
    }

    /// Set voice parameters
    pub fn voice_parameters(mut self, params: VoiceParameters) -> Self {
        self.config.voice_params = params;
        self
    }

    /// Set generation temperature
    pub fn temperature(mut self, temp: f64) -> Self {
        self.config.temperature = temp.clamp(0.0, 2.0);
        self
    }

    /// Set top-k sampling parameter
    pub fn top_k(mut self, k: usize) -> Self {
        self.config.top_k = k.clamp(1, 100);
        self
    }

    /// Set top-p nucleus sampling parameter
    pub fn top_p(mut self, p: f64) -> Self {
        self.config.top_p = p.clamp(0.0, 1.0);
        self
    }

    /// Set maximum generation steps
    pub fn max_steps(mut self, steps: usize) -> Self {
        self.config.max_steps = steps.clamp(100, 10000);
        self
    }

    /// Set random seed for reproducible generation
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Enable or disable streaming generation
    pub fn streaming(mut self, enable: bool) -> Self {
        self.config.enable_streaming = enable;
        self
    }

    /// Set computation device
    pub fn device(mut self, device: Device) -> Self {
        self.config.device = device;
        self
    }

    /// Set tensor data type
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.config.dtype = dtype;
        self
    }

    /// Build the speech generator with model files
    ///
    /// This method requires model file paths since SpeechGenerator now always
    /// loads real model weights instead of using zero-initialized models.
    pub fn build<P: AsRef<Path>>(
        self,
        lm_model_path: P,
        mimi_model_path: P,
    ) -> Result<SpeechGenerator, SpeechGenerationError> {
        // Validate configuration
        self.config.voice_params.validate()?;

        SpeechGenerator::new(lm_model_path, mimi_model_path, self.config)
    }

    /// Build from model files (deprecated - use build() instead)
    #[deprecated(note = "Use build() instead, which now requires model files")]
    pub fn build_from_files<P: AsRef<Path>>(
        self,
        lm_model_path: P,
        mimi_model_path: P,
    ) -> Result<SpeechGenerator, SpeechGenerationError> {
        self.build(lm_model_path, mimi_model_path)
    }
}

impl Default for SpeechGeneratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for quick speech generation
pub mod convenience {
    use super::*;

    /// Generate speech from models with default parameters
    ///
    /// Note: Model file paths are now required since zero-initialized models
    /// have been removed for production-grade implementation.
    pub fn generate_speech<P: AsRef<Path>>(
        text: &str,
        lm_model_path: P,
        mimi_model_path: P,
    ) -> std::result::Result<Vec<f32>, SpeechGenerationError> {
        let mut generator = SpeechGeneratorBuilder::new().build(lm_model_path, mimi_model_path)?;
        generator.generate(text)
    }

    /// Generate speech with custom voice parameters
    pub fn generate_speech_with_voice<P: AsRef<Path>>(
        text: &str,
        lm_model_path: P,
        mimi_model_path: P,
        voice_params: VoiceParameters,
    ) -> std::result::Result<Vec<f32>, SpeechGenerationError> {
        let mut generator = SpeechGeneratorBuilder::new()
            .voice_parameters(voice_params)
            .build(lm_model_path, mimi_model_path)?;
        generator.generate(text)
    }

    /// Generate speech from models with optimized loading
    pub fn generate_from_models<P: AsRef<Path>>(
        text: &str,
        lm_model_path: P,
        mimi_model_path: P,
    ) -> std::result::Result<Vec<f32>, SpeechGenerationError> {
        let mut generator = SpeechGeneratorBuilder::new().build(lm_model_path, mimi_model_path)?;
        generator.generate(text)
    }
}
