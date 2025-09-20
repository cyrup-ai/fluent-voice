//! Builder pattern for speech generator configuration

use super::{
    config::GeneratorConfig, core_generator::SpeechGenerator, error::SpeechGenerationError,
    voice_params::VoiceParameters,
};
use candle_core::{DType, Device};
use std::path::Path;

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

    /// Build the speech generator using thread-safe singleton with automatic downloading
    ///
    /// This method uses the singleton pattern to automatically download and cache models.
    /// Recommended for production use as it ensures efficient model sharing across instances.
    pub async fn build_with_download(self) -> Result<SpeechGenerator, SpeechGenerationError> {
        // Validate configuration
        self.config.voice_params.validate()?;

        SpeechGenerator::new_with_download(self.config).await
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
