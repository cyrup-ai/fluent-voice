//! Error types for the Moshi language model system

use std::fmt;

/// Main error type for Moshi operations
#[derive(Debug)]
pub enum MoshiError {
    /// Candle framework errors
    Candle(candle::Error),
    /// Configuration errors
    Config(String),
    /// Model loading errors
    ModelLoad(String),
    /// Audio processing errors
    Audio(String),
    /// Generation errors
    Generation(String),
    /// I/O errors
    Io(std::io::Error),
    /// Serialization errors
    Serde(serde_json::Error),
    /// Tokenization errors (loading, encoding, decoding)
    Tokenization(String),
    /// Custom error messages
    Custom(String),
}

impl fmt::Display for MoshiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MoshiError::Candle(e) => write!(f, "Candle error: {}", e),
            MoshiError::Config(msg) => write!(f, "Configuration error: {}", msg),
            MoshiError::ModelLoad(msg) => write!(f, "Model loading error: {}", msg),
            MoshiError::Audio(msg) => write!(f, "Audio processing error: {}", msg),
            MoshiError::Custom(msg) => write!(f, "Custom error: {}", msg),
            MoshiError::Generation(msg) => write!(f, "Generation error: {}", msg),
            MoshiError::Tokenization(msg) => write!(f, "Tokenization error: {}", msg),
            MoshiError::Io(e) => write!(f, "I/O error: {}", e),
            MoshiError::Serde(e) => write!(f, "Serialization error: {}", e),
        }
    }
}

impl std::error::Error for MoshiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MoshiError::Candle(e) => Some(e),
            MoshiError::Io(e) => Some(e),
            MoshiError::Serde(e) => Some(e),
            _ => None,
        }
    }
}

impl From<candle::Error> for MoshiError {
    fn from(error: candle::Error) -> Self {
        MoshiError::Candle(error)
    }
}

impl From<MoshiError> for candle_core::Error {
    fn from(error: MoshiError) -> Self {
        match error {
            MoshiError::Candle(e) => e,
            _ => candle_core::Error::Msg(error.to_string()),
        }
    }
}

// Note: LogitsProcessor errors are handled via manual conversion in generator.rs

impl From<std::io::Error> for MoshiError {
    fn from(error: std::io::Error) -> Self {
        MoshiError::Io(error)
    }
}

impl From<serde_json::Error> for MoshiError {
    fn from(error: serde_json::Error) -> Self {
        MoshiError::Serde(error)
    }
}

/// Result type alias for Moshi operations
pub type Result<T> = std::result::Result<T, MoshiError>;
