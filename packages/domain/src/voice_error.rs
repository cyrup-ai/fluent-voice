//! Unified error for all fluent chains.
use thiserror::Error;

/// Top-level error covering both TTS & STT operations.
#[derive(Debug, Error)]
pub enum VoiceError {
    /// TTS-related failure reason.
    #[error("tts: {0}")]
    Tts(&'static str),
    /// STT-related failure reason.
    #[error("stt: {0}")]
    Stt(&'static str),
    /// Configuration-related failure reason.
    #[error("configuration: {0}")]
    ConfigurationError(String),
    /// Processing-related failure reason.
    #[error("processing: {0}")]
    ProcessingError(String),
}
