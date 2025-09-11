//! Speaker builder trait for fluent voice API.
//!
//! This module defines the SpeakerBuilder trait that provides a fluent interface
//! for creating and configuring speaker objects with voice settings.

use fluent_voice_domain::{Language, PitchRange, VocalSpeedMod, VoiceId};

/// Builder trait for creating speaker configurations.
///
/// This trait provides a fluent API for constructing speaker objects with
/// various voice characteristics like voice ID, language, speed modifiers,
/// and pitch ranges.
pub trait SpeakerBuilder: Sized + Send {
    /// The concrete speaker type produced by this builder.
    type Output: Send;

    /// Create a new speaker builder with the given name/identifier.
    fn speaker(name: impl Into<String>) -> Self;

    /// Set the voice ID for this speaker.
    fn voice_id(self, id: VoiceId) -> Self;

    /// Set the language for this speaker.
    fn language(self, lang: Language) -> Self;

    /// Add a prelude text that will be spoken before the main content.
    fn with_prelude(self, prelude: impl Into<String>) -> Self;

    /// Add a line of text to be spoken.
    fn add_line(self, line: impl Into<String>) -> Self;

    /// Set the voice using a string identifier.
    fn with_voice(self, voice: impl Into<String>) -> Self;

    /// Set the speaking speed as a float multiplier.
    fn with_speed(self, speed: f32) -> Self;

    /// Set the speed modifier using the domain type.
    fn with_speed_modifier(self, modifier: VocalSpeedMod) -> Self;

    /// Set the pitch range for this speaker.
    fn with_pitch_range(self, range: PitchRange) -> Self;

    /// Set the main text to be spoken.
    fn speak(self, text: impl Into<String>) -> Self;

    /// Build the final speaker object.
    fn build(self) -> Self::Output;
}
