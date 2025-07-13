//! Speaker and SpeakerBuilder traits

use super::{VoicePersona, VoiceTimber};
use std::path::Path;

/// Trait for speaker builders
pub trait SpeakerBuilder: Sized {
    fn with_clone_from_path(self, path: impl AsRef<Path>) -> Self;
    fn with_timber(self, timber: VoiceTimber) -> Self;
    fn with_persona_trait(self, persona: VoicePersona) -> Self;
}

/// Trait for fully configured speakers
pub trait Speaker: Send + Sync {
    /// Create a new speaker builder with the given name
    fn named(name: impl Into<String>) -> impl SpeakerBuilder;

    /// Get this speaker's ID
    fn id(&self) -> &str;

    /// Get the voice clone for audio prompting (optional)
    fn voice_clone(&self) -> Option<&super::VoiceClone> {
        None
    }
}
