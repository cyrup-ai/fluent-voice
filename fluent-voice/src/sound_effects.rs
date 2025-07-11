//! Sound effects generation builder.

use crate::audio_format::AudioFormat;
use core::future::Future;
use fluent_voice_domain::VoiceError;
use futures_core::Stream;

/// Sound effects generation session.
///
/// This trait represents a configured sound effects generation operation
/// that creates audio effects from text descriptions using AI.
pub trait SoundEffectsSession: Send {
    /// Audio stream type that will be produced.
    type AudioStream: Stream<Item = i16> + Send + Unpin;

    /// Convert this session into an audio stream.
    ///
    /// This method consumes the session and returns the underlying
    /// audio stream containing the generated sound effects.
    fn into_stream(self) -> Self::AudioStream;
}

/// Fluent builder for sound effects generation.
///
/// This trait provides the interface for generating audio effects
/// from text descriptions using AI sound synthesis.
pub trait SoundEffectsBuilder: Sized + Send {
    /// The session type produced by this builder.
    type Session: SoundEffectsSession;

    /// Set the description of the sound effect to generate.
    ///
    /// The description should be detailed and specific about the
    /// desired sound characteristics, environment, and mood.
    ///
    /// # Arguments
    ///
    /// * `description` - Text description of the desired sound effect
    ///
    /// # Examples
    ///
    /// ```ignore
    /// .describe("thunderstorm with heavy rain and distant thunder")
    /// .describe("bustling city street with car traffic and pedestrians")
    /// .describe("peaceful forest with birds chirping and wind in trees")
    /// ```
    fn describe(self, description: impl Into<String>) -> Self;

    /// Set the duration of the generated sound effect.
    ///
    /// # Arguments
    ///
    /// * `seconds` - Duration in seconds (typically 1-60 seconds)
    fn duration_seconds(self, seconds: f32) -> Self;

    /// Set the intensity or volume level of the effect.
    ///
    /// Controls how prominent or subtle the generated sound effect
    /// should be. Higher values create more intense, dramatic effects.
    ///
    /// # Arguments
    ///
    /// * `intensity` - Intensity level between 0.0 and 1.0
    fn intensity(self, intensity: f32) -> Self;

    /// Set the mood or emotional tone of the effect.
    ///
    /// # Arguments
    ///
    /// * `mood` - Descriptive mood (e.g., "peaceful", "dramatic", "eerie")
    fn mood(self, mood: impl Into<String>) -> Self;

    /// Set the environmental context for the effect.
    ///
    /// Helps the AI understand the spatial and environmental
    /// characteristics of the desired sound.
    ///
    /// # Arguments
    ///
    /// * `environment` - Environmental context (e.g., "indoor", "outdoor", "urban", "nature")
    fn environment(self, environment: impl Into<String>) -> Self;

    /// Set the output audio format.
    ///
    /// # Arguments
    ///
    /// * `format` - Desired output audio format
    fn output_format(self, format: AudioFormat) -> Self;

    /// Set a seed for deterministic generation.
    ///
    /// When set, the same description and parameters will always
    /// produce the same sound effect output.
    ///
    /// # Arguments
    ///
    /// * `seed` - Deterministic seed value
    fn seed(self, seed: u64) -> Self;

    /// Terminal method that executes sound generation with a matcher closure.
    ///
    /// This method terminates the fluent chain and executes the sound effects
    /// generation. The matcher closure receives either the session object on success
    /// or a `VoiceError` on failure, and returns the final result.
    ///
    /// # Arguments
    ///
    /// * `matcher` - Closure that handles success/error cases
    ///
    /// # Returns
    ///
    /// A future that resolves to the result of the matcher closure.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let audio = FluentVoice::sound_effects()
    ///     .describe("thunderstorm with heavy rain")
    ///     .duration_seconds(30.0)
    ///     .intensity(0.8)
    ///     .mood("dramatic")
    ///     .generate(|session| {
    ///         Ok => session.into_stream(),
    ///         Err(e) => Err(e),
    ///     })
    ///     .await?;
    /// ```
    fn generate<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R + Send + 'static;
}

/// Static entry point for sound effects generation.
///
/// This trait provides the static method for starting sound effects
/// generation operations. Engine implementations typically implement
/// this on a marker struct or their main engine type.
///
/// # Examples
///
/// ```ignore
/// use fluent_voice::prelude::*;
///
/// let effects = MyEngine::sound_effects();
/// ```
pub trait SoundEffectsExt {
    /// Begin a new sound effects generation builder.
    ///
    /// # Returns
    ///
    /// A new sound effects builder instance.
    fn builder() -> impl SoundEffectsBuilder;
}
