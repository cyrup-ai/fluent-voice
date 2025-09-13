//! Wake word detection error handling.

use std::fmt;

/// Errors that can occur during wake word detection.
#[derive(Debug, Clone, PartialEq)]
pub enum WakeWordError {
    /// Audio chunk size doesn't match expected frame size.
    WrongChunk {
        /// Expected chunk size in samples.
        expected: usize,
        /// Actual chunk size received.
        got: usize,
    },

    /// Ring buffer overflow - position exceeded buffer bounds.
    RingOverflow,

    /// Model loading failed.
    ModelLoadFailed {
        /// Description of the failure.
        reason: String,
    },

    /// Feature extraction failed.
    FeatureExtractionFailed {
        /// Description of the failure.
        reason: String,
    },

    /// Invalid configuration parameters.
    InvalidConfig {
        /// Description of the invalid configuration.
        reason: String,
    },

    /// Voice Activity Detection failed.
    VadFailed {
        /// Description of the VAD failure.
        reason: String,
    },
}

impl fmt::Display for WakeWordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WakeWordError::WrongChunk { expected, got } => {
                write!(
                    f,
                    "Audio chunk size mismatch: expected {} samples, got {}",
                    expected, got
                )
            }
            WakeWordError::RingOverflow => {
                write!(f, "Ring buffer overflow: position exceeded buffer bounds")
            }
            WakeWordError::ModelLoadFailed { reason } => {
                write!(f, "Model loading failed: {}", reason)
            }
            WakeWordError::FeatureExtractionFailed { reason } => {
                write!(f, "Feature extraction failed: {}", reason)
            }
            WakeWordError::InvalidConfig { reason } => {
                write!(f, "Invalid configuration: {}", reason)
            }
            WakeWordError::VadFailed { reason } => {
                write!(f, "Voice Activity Detection failed: {}", reason)
            }
        }
    }
}

impl std::error::Error for WakeWordError {}

impl From<Box<dyn std::error::Error>> for WakeWordError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        WakeWordError::VadFailed {
            reason: err.to_string(),
        }
    }
}

/// Result type for wake word operations.
pub type Result<T> = std::result::Result<T, WakeWordError>;
