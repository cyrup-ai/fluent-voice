//! Unified error for all fluent chains.
use thiserror::Error;

/// Top-level error covering both TTS & STT operations.
#[derive(Debug, Clone, Error)]
pub enum VoiceError {
    /// TTS-related failure reason.
    #[error("tts: {0}")]
    Tts(&'static str),
    /// STT-related failure reason.
    #[error("stt: {0}")]
    Stt(&'static str),
    /// Configuration-related failure reason.
    #[error("configuration: {0}")]
    Configuration(String),
    /// Processing-related failure reason.
    #[error("processing: {0}")]
    ProcessingError(String),
    /// Error during synthesis operation
    #[error("synthesis: {0}")]
    Synthesis(String),
    /// Error when synthesis is not possible
    #[error("not synthesizable: {0}")]
    NotSynthesizable(String),
    /// Error during transcription operation
    #[error("transcription: {0}")]
    Transcription(String),
    /// Error during audio processing operation
    #[error("audio processing: {0}")]
    AudioProcessing(String),
}
