//! Immutable fluent builder for one speaker turn.
use crate::speaker::Speaker;
use fluent_voice_domain::{Language, PitchRange, VocalSpeedMod, VoiceId};

/// Immutable fluent builder for configuring a single speaker turn.
///
/// This trait provides a fluent interface for building speaker configurations
/// with voice settings, text content, and expressive parameters.
pub trait SpeakerBuilder: Sized {
    /// Start a builder with a display name.
    fn speaker(name: impl Into<String>) -> Self;

    /// Associate an engine-specific voice ID.
    fn voice_id(self, id: VoiceId) -> Self;

    /// Override language for this speaker.
    fn language(self, lang: Language) -> Self;

    /// Optional speaking-rate multiplier.
    fn with_speed_modifier(self, m: VocalSpeedMod) -> Self;

    /// Optional pitch range.
    fn with_pitch_range(self, range: PitchRange) -> Self;

    /// Provide text for this speaker to speak.
    fn speak(self, text: impl Into<String>) -> Self;

    /// The concrete speaker type that will be produced.
    type Output: Speaker;

    /// Finish and get the concrete `Speaker`.
    fn build(self) -> Self::Output;
}

/// Convenience extension for creating speakers.
///
/// Allows syntax like `Speaker::speaker("Bob")` to start building.
pub trait SpeakerExt {
    /// Start a fluent speaker builder.
    fn speaker(name: impl Into<String>) -> impl SpeakerBuilder {
        Self::SpeakerType::speaker(name)
    }

    /// The concrete speaker builder type this extension provides.
    type SpeakerType: SpeakerBuilder;
}
