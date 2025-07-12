//! Runtime speaker instance (engine provides concrete type).

use fluent_voice_domain::{Language, PitchRange, VocalSpeedMod, VoiceId};
use crate::speaker_builder::SpeakerBuilder;

/// Runtime speaker handle that represents a configured voice with text to speak.
///
/// This trait is implemented by engine-specific speaker types that contain
/// all the configuration needed for a single speaking turn (voice ID, text,
/// speed modifiers, etc.).
pub trait Speaker: Clone + Send + Sync {
    /// Returns a unique identifier for this speaker instance.
    ///
    /// Used primarily for debugging and logging purposes.
    fn id(&self) -> &str;

    /// Returns the text to be spoken by this speaker.
    fn text(&self) -> &str;

    /// Returns the voice ID to use for this speaker, if specified.
    fn voice_id(&self) -> Option<&VoiceId>;

    /// Returns the language override for this speaker, if specified.
    fn language(&self) -> Option<&Language>;

    /// Returns the speed modifier for this speaker, if specified.
    fn speed_modifier(&self) -> Option<VocalSpeedMod>;

    /// Returns the pitch range for this speaker, if specified.
    fn pitch_range(&self) -> Option<&PitchRange>;
}

/// Implementation of SpeakerExt for all Speaker types
impl<T> crate::speaker_builder::SpeakerExt for T
where
    T: Speaker,
{
    fn speaker(name: impl Into<String>) -> impl crate::speaker_builder::SpeakerBuilder {
        <crate::builders::SpeakerLineBuilder as SpeakerBuilder>::named(name)
    }
}
