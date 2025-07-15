//! Wake word detection configuration.

use serde::{Deserialize, Serialize};

/// Configuration for wake word detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    /// Voice Activity Detection threshold (0.0 to 1.0).
    /// Higher values make VAD more selective.
    pub vad_threshold: f32,

    /// Wake word detection threshold (0.0 to 1.0).
    /// Higher values reduce false positives but may miss wake words.
    pub wake_threshold: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vad_threshold: 0.5,
            wake_threshold: 0.7,
        }
    }
}

impl Config {
    /// Create a new configuration with custom thresholds.
    pub fn new(vad_threshold: f32, wake_threshold: f32) -> Self {
        Self {
            vad_threshold,
            wake_threshold,
        }
    }

    /// Create a configuration optimized for low false positives.
    /// Uses higher thresholds to reduce spurious activations.
    pub fn conservative() -> Self {
        Self {
            vad_threshold: 0.7,
            wake_threshold: 0.8,
        }
    }

    /// Create a configuration optimized for high sensitivity.
    /// Uses lower thresholds to catch more wake words.
    pub fn sensitive() -> Self {
        Self {
            vad_threshold: 0.3,
            wake_threshold: 0.5,
        }
    }
}
