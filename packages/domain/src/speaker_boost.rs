//! Speaker boost parameter.
//!
//! Controls how distinctly different speakers are emphasized
//! in multi-speaker conversations.

/// Speaker boost setting (enable/disable).
#[derive(Clone, Debug)]
pub struct SpeakerBoost(bool);

impl SpeakerBoost {
    /// Create a new speaker boost setting.
    ///
    /// When enabled (true), the TTS engine will enhance the distinction
    /// between different speakers in multi-speaker conversations.
    pub fn new(enabled: bool) -> Self {
        Self(enabled)
    }

    /// Check if speaker boost is enabled.
    pub fn is_enabled(&self) -> bool {
        self.0
    }
}

impl Default for SpeakerBoost {
    fn default() -> Self {
        Self(true) // Default to enabled
    }
}
