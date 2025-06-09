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
}
