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
    /// LiveKit integration errors
    LiveKit(String),
    /// Audio bridge errors  
    AudioBridge(String),
    /// I/O errors
    Io(std::io::Error),
    /// Serialization errors
    Serde(serde_json::Error),
    /// Tokenization errors (loading, encoding, decoding)
    Tokenization(String),
    /// Custom error messages
    Custom(String),
    /// Mutex poisoning error with context about which mutex failed
    MutexPoisoned(String),
    /// State corruption detected due to concurrent access failures  
    StateCorruption(String),
    /// Device operation failed with context
    DeviceError(String),
    /// Tensor creation failed with detailed context  
    TensorCreationError(String),
    /// Empty vector access attempted
    EmptyVectorAccess(String),
    /// Vector size validation failed
    VectorSizeValidation(String),
    /// Embedding dimension mismatch
    EmbeddingDimensionMismatch { expected: Vec<usize>, actual: Vec<usize> },
    /// Tensor concatenation failed
    TensorConcatenationError(String),
    /// Shape validation failed
    ShapeMismatch(String),
    /// Input validation failed
    InvalidInput(String),
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
            MoshiError::LiveKit(msg) => write!(f, "LiveKit integration error: {}", msg),
            MoshiError::AudioBridge(msg) => write!(f, "Audio bridge error: {}", msg),
            MoshiError::Tokenization(msg) => write!(f, "Tokenization error: {}", msg),
            MoshiError::Io(e) => write!(f, "I/O error: {}", e),
            MoshiError::Serde(e) => write!(f, "Serialization error: {}", e),
            MoshiError::MutexPoisoned(msg) => write!(f, "Mutex poisoning error: {}", msg),
            MoshiError::StateCorruption(msg) => write!(f, "State corruption error: {}", msg),
            MoshiError::DeviceError(msg) => write!(f, "Device operation failed: {}", msg),
            MoshiError::TensorCreationError(msg) => write!(f, "Tensor creation failed: {}", msg),
            MoshiError::EmptyVectorAccess(msg) => write!(f, "Empty vector access attempted: {}", msg),
            MoshiError::VectorSizeValidation(msg) => write!(f, "Vector size validation failed: {}", msg),
            MoshiError::EmbeddingDimensionMismatch { expected, actual } => {
                write!(f, "Embedding dimension mismatch: expected {:?}, got {:?}", expected, actual)
            }
            MoshiError::TensorConcatenationError(msg) => write!(f, "Tensor concatenation failed: {}", msg),
            MoshiError::ShapeMismatch(msg) => write!(f, "Tensor shape mismatch: {}", msg),
            MoshiError::InvalidInput(msg) => write!(f, "Invalid input provided: {}", msg),
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

impl From<crate::speech_generator::error::SpeechGenerationError> for MoshiError {
    fn from(error: crate::speech_generator::error::SpeechGenerationError) -> Self {
        MoshiError::Generation(error.to_string())
    }
}

impl From<crate::livekit_bridge::LiveKitBridgeError> for MoshiError {
    fn from(error: crate::livekit_bridge::LiveKitBridgeError) -> Self {
        match error {
            crate::livekit_bridge::LiveKitBridgeError::ConnectionError(msg) => MoshiError::LiveKit(msg),
            crate::livekit_bridge::LiveKitBridgeError::RoomError(msg) => MoshiError::LiveKit(msg),
            crate::livekit_bridge::LiveKitBridgeError::MicrophoneError(msg) => MoshiError::LiveKit(msg),
            crate::livekit_bridge::LiveKitBridgeError::AudioConversionError(msg) => MoshiError::AudioBridge(msg),
            crate::livekit_bridge::LiveKitBridgeError::CommunicationError(msg) => MoshiError::AudioBridge(msg),
        }
    }
}

/// Result type alias for Moshi operations
pub type Result<T> = std::result::Result<T, MoshiError>;

/// Context-rich error creation helper
pub fn create_tensor_error_context(
    operation: &str,
    shape: &candle_core::Shape,
    dtype: candle_core::DType,
    device: &candle_core::Device,
    error: candle_core::Error,
) -> crate::error::MoshiError {
    crate::error::MoshiError::TensorCreationError(format!(
        "Operation '{}' failed: shape={:?}, dtype={:?}, device={:?}, error={}",
        operation, shape, dtype, device, error
    ))
}
