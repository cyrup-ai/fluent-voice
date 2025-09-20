//! Error types for speech generation

use crate::error::MoshiError;

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
            MoshiError::MutexPoisoned(msg) => {
                SpeechGenerationError::AudioGeneration(format!("Mutex poisoning: {}", msg))
            }
            MoshiError::StateCorruption(msg) => {
                SpeechGenerationError::AudioGeneration(format!("State corruption: {}", msg))
            }
        }
    }
}

impl From<candle_core::Error> for SpeechGenerationError {
    fn from(err: candle_core::Error) -> Self {
        SpeechGenerationError::TensorOperation(err.to_string())
    }
}
