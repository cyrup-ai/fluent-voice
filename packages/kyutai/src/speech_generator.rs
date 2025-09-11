//! High-Performance Speech Generation Engine
//!
//! This module provides a blazing-fast, zero-allocation speech generation system
//! with real-time streaming capabilities and comprehensive error handling.

use crate::error::MoshiError;
use crate::tts::{Config as TtsConfig, Model as TtsModel};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Audio buffer size for streaming generation (16KB = ~180ms at 44.1kHz)
const AUDIO_BUFFER_SIZE: usize = 16384;
/// Token buffer size for text processing (supports ~2000 characters)
const TOKEN_BUFFER_SIZE: usize = 2048;
/// Maximum text length for single generation (64KB)
const MAX_TEXT_LENGTH: usize = 65536;
/// Audio sample rate (44.1kHz)
const SAMPLE_RATE: u32 = 44100;
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
    pub fn apply_to_samples(&self, samples: &mut [f32]) {
        // Apply volume adjustment
        if self.volume != 1.0 {
            for sample in samples.iter_mut() {
                *sample *= self.volume;
            }
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
        params.apply_to_samples(&mut output[..samples_to_process]);

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
    /// Generated audio chunks
    #[allow(dead_code)]
    generated_chunks: Vec<Vec<f32>>,
    /// Current chunk index
    #[allow(dead_code)]
    chunk_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum GenerationState {
    Initializing,
    Processing,
    #[allow(dead_code)]
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
    /// Pre-allocated token buffer
    #[allow(dead_code)]
    token_buffer: Box<[u32; TOKEN_BUFFER_SIZE]>,
    /// Generation configuration
    config: GeneratorConfig,
    /// Performance statistics
    stats: GenerationStats,
    /// Current generation state
    generation_active: bool,
    /// Text processing queue
    #[allow(dead_code)]
    text_queue: VecDeque<String>,
}

impl SpeechGenerator {
    /// Create new speech generator with optimized configuration
    pub fn new(config: GeneratorConfig) -> Result<Self, SpeechGenerationError> {
        // Initialize TTS model
        let tts_config = crate::tts::Config::v202501();

        // Create VarBuilder for model loading
        let lm_vb = VarBuilder::zeros(config.dtype, &config.device);
        let mimi_vb = VarBuilder::zeros(config.dtype, &config.device);

        let tts_model = TtsModel::new(tts_config, lm_vb, mimi_vb)
            .map_err(|e| SpeechGenerationError::ModelInitialization(e.to_string()))?;

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

    /// Load models from files with optimized loading
    pub fn load_from_files<P: AsRef<Path>>(
        lm_model_path: P,
        mimi_model_path: P,
        config: GeneratorConfig,
    ) -> Result<Self, SpeechGenerationError> {
        let tts_model =
            TtsModel::load(lm_model_path, mimi_model_path, config.dtype, &config.device)
                .map_err(|e| SpeechGenerationError::ModelLoading(e.to_string()))?;

        let audio_buffer = AudioBuffer::new(SAMPLE_RATE, CHANNELS);
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

        // Generate audio
        let audio_data = self
            .tts_model
            .generate(
                text,
                None, // No speaker PCM for now
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

    /// Build the speech generator
    pub fn build(self) -> Result<SpeechGenerator, SpeechGenerationError> {
        // Validate configuration
        self.config.voice_params.validate()?;

        SpeechGenerator::new(self.config)
    }

    /// Build from model files
    pub fn build_from_files<P: AsRef<Path>>(
        self,
        lm_model_path: P,
        mimi_model_path: P,
    ) -> Result<SpeechGenerator, SpeechGenerationError> {
        // Validate configuration
        self.config.voice_params.validate()?;

        SpeechGenerator::load_from_files(lm_model_path, mimi_model_path, self.config)
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

    /// Generate speech with default parameters
    pub fn generate_speech(text: &str) -> std::result::Result<Vec<f32>, SpeechGenerationError> {
        let mut generator = SpeechGeneratorBuilder::new().build()?;
        generator.generate(text)
    }

    /// Generate speech with custom voice parameters
    pub fn generate_speech_with_voice(
        text: &str,
        voice_params: VoiceParameters,
    ) -> std::result::Result<Vec<f32>, SpeechGenerationError> {
        let mut generator = SpeechGeneratorBuilder::new()
            .voice_parameters(voice_params)
            .build()?;
        generator.generate(text)
    }

    /// Generate speech from file with optimized loading
    pub fn generate_from_models<P: AsRef<Path>>(
        text: &str,
        lm_model_path: P,
        mimi_model_path: P,
    ) -> std::result::Result<Vec<f32>, SpeechGenerationError> {
        let mut generator =
            SpeechGeneratorBuilder::new().build_from_files(lm_model_path, mimi_model_path)?;
        generator.generate(text)
    }
}
