//! fluent_voice/src/voice_error.rs
//! -------------------------------
//! Voice synthesis error types

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VoiceError {
    #[error("configuration: {0}")]
    Config(&'static str),
    #[error("device error: {0}")]
    Device(&'static str),
    #[error("synthesis failed: {0}")]
    Synthesis(&'static str),
}