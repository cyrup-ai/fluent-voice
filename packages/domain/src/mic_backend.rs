//! Microphone device selection.
use serde::{Deserialize, Serialize};

/// Microphone backend/device selection for audio capture.
///
/// This enum allows specifying which microphone device to use for
/// speech-to-text input. Different engines may interpret device
/// identifiers differently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MicBackend {
    /// Use the system's default microphone device.
    Default,
    /// Use a specific microphone device by name or identifier.
    Device(String),
}
