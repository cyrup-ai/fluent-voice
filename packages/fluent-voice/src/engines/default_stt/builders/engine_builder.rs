//! Engine Builder for Default STT Engine

use crate::engines::default_stt::{DefaultSTTEngine, VadConfig, WakeWordConfig};
use fluent_voice_domain::VoiceError;

/// Builder for configuring the default STT engine with fluent API.
pub struct DefaultSTTEngineBuilder {
    vad_config: VadConfig,
    wake_word_config: WakeWordConfig,
}

impl DefaultSTTEngineBuilder {
    /// Create a new default STT engine builder.
    pub fn new() -> Self {
        Self {
            vad_config: VadConfig::default(),
            wake_word_config: WakeWordConfig::default(),
        }
    }

    /// Configure VAD sensitivity (0.0 to 1.0).
    pub fn with_vad_sensitivity(mut self, sensitivity: f32) -> Self {
        self.vad_config.sensitivity = sensitivity;
        self
    }

    /// Configure minimum speech duration in milliseconds.
    pub fn with_min_speech_duration(mut self, duration: u32) -> Self {
        self.vad_config.min_speech_duration = duration;
        self
    }

    /// Configure maximum silence duration in milliseconds.
    pub fn with_max_silence_duration(mut self, duration: u32) -> Self {
        self.vad_config.max_silence_duration = duration;
        self
    }

    /// Configure wake word model (default: "syrup").
    /// Note: Model selection is handled during detector initialization, not in config.
    pub fn with_wake_word_model<S: Into<String>>(self, _model: S) -> Self {
        // WakeWordConfig doesn't store model name - it's handled during detector creation
        self
    }

    /// Configure wake word detection threshold (0.0 to 1.0).
    pub fn with_wake_word_threshold(mut self, threshold: f32) -> Self {
        self.wake_word_config.sensitivity = threshold;
        self
    }

    /// Build the configured default STT engine.
    pub async fn build(self) -> Result<DefaultSTTEngine, VoiceError> {
        DefaultSTTEngine::with_config(self.vad_config, self.wake_word_config).await
    }
}

impl Default for DefaultSTTEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
