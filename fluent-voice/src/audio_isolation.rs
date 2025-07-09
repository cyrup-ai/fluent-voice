//! Audio isolation and processing builder.

use crate::{
    audio_format::AudioFormat,
    voice_error::VoiceError,
};
use core::future::Future;
use futures_core::Stream;

/// Audio isolation session.
///
/// This trait represents a configured audio isolation operation that
/// separates voices from background noise or isolates specific audio
/// components from mixed audio content.
pub trait AudioIsolationSession: Send {
    /// Audio stream type that will be produced.
    type AudioStream: Stream<Item = i16> + Send + Unpin;

    /// Convert this session into an audio stream.
    ///
    /// This method consumes the session and returns the underlying
    /// audio stream containing the isolated audio.
    fn into_stream(self) -> Self::AudioStream;
}

/// Fluent builder for audio isolation operations.
///
/// This trait provides the interface for separating voices from background
/// audio, removing noise, or isolating specific audio components.
pub trait AudioIsolationBuilder: Sized + Send {
    /// The session type produced by this builder.
    type Session: AudioIsolationSession;

    /// Set the input audio source.
    ///
    /// This can be a file path containing mixed audio to be processed.
    ///
    /// # Arguments
    ///
    /// * `source` - Path to input audio file
    fn from_file(self, source: impl Into<String>) -> Self;

    /// Set the input audio data directly.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw audio data bytes
    fn from_audio_data(self, data: Vec<u8>) -> Self;

    /// Enable voice isolation.
    ///
    /// When enabled, attempts to isolate human speech from the audio
    /// while removing background noise and other audio elements.
    ///
    /// # Arguments
    ///
    /// * `isolate` - Whether to isolate voices
    fn isolate_voices(self, isolate: bool) -> Self;

    /// Enable background removal.
    ///
    /// When enabled, removes background music, ambient noise, and
    /// other non-speech audio elements.
    ///
    /// # Arguments
    ///
    /// * `remove` - Whether to remove background audio
    fn remove_background(self, remove: bool) -> Self;

    /// Enable noise reduction.
    ///
    /// When enabled, applies advanced noise reduction algorithms
    /// to improve audio clarity.
    ///
    /// # Arguments
    ///
    /// * `reduce` - Whether to apply noise reduction
    fn reduce_noise(self, reduce: bool) -> Self;

    /// Set the isolation strength.
    ///
    /// Controls how aggressively the isolation algorithm separates
    /// different audio components. Higher values provide stronger
    /// separation but may introduce artifacts.
    ///
    /// # Arguments
    ///
    /// * `strength` - Isolation strength between 0.0 and 1.0
    fn isolation_strength(self, strength: f32) -> Self;

    /// Set the output audio format.
    ///
    /// # Arguments
    ///
    /// * `format` - Desired output audio format
    fn output_format(self, format: AudioFormat) -> Self;

    /// Terminal method that executes audio isolation with a matcher closure.
    ///
    /// This method terminates the fluent chain and executes the audio isolation.
    /// The matcher closure receives either the session object on success
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
    /// let audio = FluentVoice::audio_isolation()
    ///     .from_file("mixed_audio.wav")
    ///     .isolate_voices(true)
    ///     .remove_background(true)
    ///     .process(|session| {
    ///         Ok => session.into_stream(),
    ///         Err(e) => Err(e),
    ///     })
    ///     .await?;
    /// ```
    fn process<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R + Send + 'static;
}

/// Static entry point for audio isolation.
///
/// This trait provides the static method for starting audio isolation
/// operations. Engine implementations typically implement this on a
/// marker struct or their main engine type.
///
/// # Examples
///
/// ```ignore
/// use fluent_voice::prelude::*;
///
/// let isolation = MyEngine::audio_isolation();
/// ```
pub trait AudioIsolationExt {
    /// Begin a new audio isolation builder.
    ///
    /// # Returns
    ///
    /// A new audio isolation builder instance.
    fn builder() -> impl AudioIsolationBuilder;
}