//! Voice-activity / turn detection modes.
use serde::{Deserialize, Serialize};

/// Voice activity detection and turn detection aggressiveness.
///
/// Controls how the STT engine detects speech boundaries and
/// determines when a speaker has finished talking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VadMode {
    /// Disable VAD entirely (continuous transcription).
    Off,
    /// Low-latency, less accurate end-of-speech detection.
    Fast,
    /// More accurate, possibly higher latency detection.
    Accurate,
}
