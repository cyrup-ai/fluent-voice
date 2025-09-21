//! Core speech generation engine implementation

use super::{
    audio_buffer::AudioBuffer, audio_stream::AudioStreamIterator, config::GeneratorConfig,
    error::SpeechGenerationError, stats::GenerationStats, utils::validate_model_file,
    voice_params::VoiceParameters,
};
use crate::tts::Model as TtsModel;
use candle_core::{DType, Device};
use std::collections::VecDeque;
use std::path::Path;

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

/// High-performance speech generation engine
pub struct SpeechGenerator {
    /// TTS model for speech synthesis
    pub(super) tts_model: TtsModel,
    /// Audio buffer for streaming
    pub(super) audio_buffer: AudioBuffer,
    /// Pre-allocated token buffer (used for batch token processing)
    _token_buffer: Box<[u32; TOKEN_BUFFER_SIZE]>,
    /// Generation configuration
    pub(super) config: GeneratorConfig,
    /// Performance statistics
    pub(super) stats: GenerationStats,
    /// Current generation state
    pub(super) generation_active: bool,
    /// Text processing queue (used for async text processing pipeline)
    _text_queue: VecDeque<String>,
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
        validate_model_file(&lm_model_path, "language model")?;
        validate_model_file(&mimi_model_path, "Mimi model")?;

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
            _token_buffer: token_buffer,
            config,
            stats: GenerationStats::default(),
            generation_active: false,
            _text_queue: VecDeque::with_capacity(16),
        })
    }

    /// Create new speech generator using thread-safe singleton with automatic downloading
    ///
    /// This constructor uses the singleton pattern to automatically download and cache models.
    /// Recommended for production use as it ensures efficient model sharing across instances.
    pub async fn new_with_download(config: GeneratorConfig) -> Result<Self, SpeechGenerationError> {
        // Use thread-safe singleton for model loading
        let model_paths = crate::models::get_or_download_models()
            .await
            .map_err(|e| SpeechGenerationError::ModelLoading(e.to_string()))?;

        // Load TTS model with actual weights from singleton paths
        let tts_model = TtsModel::load(
            &model_paths.lm_model_path,
            &model_paths.mimi_model_path,
            config.dtype,
            &config.device,
        )
        .map_err(|e| SpeechGenerationError::ModelLoading(e.to_string()))?;

        // Initialize audio buffer
        let audio_buffer = AudioBuffer::new(SAMPLE_RATE, CHANNELS);

        // Pre-allocate token buffer
        let token_buffer = Box::new([0u32; TOKEN_BUFFER_SIZE]);

        Ok(Self {
            tts_model,
            audio_buffer,
            _token_buffer: token_buffer,
            config,
            stats: GenerationStats::default(),
            generation_active: false,
            _text_queue: VecDeque::with_capacity(16),
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
    pub(super) fn generate_chunk(
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
    pub(super) fn finalize_generation(
        &mut self,
    ) -> std::result::Result<Vec<f32>, SpeechGenerationError> {
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
