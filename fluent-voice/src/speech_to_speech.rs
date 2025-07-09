//! Speech-to-speech voice conversion builder.

use crate::{
    audio_format::AudioFormat,
    model_id::ModelId,
    voice_error::VoiceError,
    voice_id::VoiceId,
};
use core::future::Future;
use futures_core::Stream;

/// Speech-to-speech conversion session.
///
/// This trait represents a configured speech-to-speech conversion that
/// converts input speech to output speech with a different voice while
/// preserving speech characteristics like emotion and timing.
pub trait SpeechToSpeechSession: Send {
    /// Audio stream type that will be produced.
    type AudioStream: Stream<Item = i16> + Send + Unpin;

    /// Convert this session into an audio stream.
    ///
    /// This method consumes the session and returns the underlying
    /// audio stream containing the converted speech.
    fn into_stream(self) -> Self::AudioStream;
}

/// Fluent builder for speech-to-speech voice conversion.
///
/// This trait provides the interface for converting speech from one voice
/// to another while preserving emotional content, timing, and prosody.
pub trait SpeechToSpeechBuilder: Sized + Send {
    /// The session type produced by this builder.
    type Session: SpeechToSpeechSession;

    /// Set the input audio source.
    ///
    /// This can be a file path or audio data to be converted.
    ///
    /// # Arguments
    ///
    /// * `source` - Path to input audio file
    fn from_audio(self, source: impl Into<String>) -> Self;

    /// Set the input audio data directly.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw audio data bytes
    fn from_audio_data(self, data: Vec<u8>) -> Self;

    /// Set the target voice for conversion.
    ///
    /// The input speech will be converted to sound like this voice.
    ///
    /// # Arguments
    ///
    /// * `voice_id` - Target voice identifier
    fn target_voice(self, voice_id: VoiceId) -> Self;

    /// Set the model to use for conversion.
    ///
    /// Different models may provide different quality/speed tradeoffs.
    ///
    /// # Arguments
    ///
    /// * `model` - Model ID for speech conversion
    fn model(self, model: ModelId) -> Self;

    /// Control whether to preserve emotional content.
    ///
    /// When enabled, the emotional characteristics of the input speech
    /// are preserved in the output.
    ///
    /// # Arguments
    ///
    /// * `preserve` - Whether to preserve emotion
    fn preserve_emotion(self, preserve: bool) -> Self;

    /// Control whether to preserve speaking style.
    ///
    /// When enabled, the speaking style and mannerisms of the input
    /// are preserved while changing the voice.
    ///
    /// # Arguments
    ///
    /// * `preserve` - Whether to preserve style
    fn preserve_style(self, preserve: bool) -> Self;

    /// Control whether to preserve timing and prosody.
    ///
    /// When enabled, the timing, rhythm, and intonation patterns
    /// of the input speech are preserved.
    ///
    /// # Arguments
    ///
    /// * `preserve` - Whether to preserve timing
    fn preserve_timing(self, preserve: bool) -> Self;

    /// Set the output audio format.
    ///
    /// # Arguments
    ///
    /// * `format` - Desired output audio format
    fn output_format(self, format: AudioFormat) -> Self;

    /// Set voice stability for the conversion.
    ///
    /// Controls how consistent the output voice characteristics remain.
    ///
    /// # Arguments
    ///
    /// * `stability` - Stability value between 0.0 and 1.0
    fn stability(self, stability: f32) -> Self;

    /// Set voice similarity boost.
    ///
    /// Controls how closely the output matches the target voice.
    ///
    /// # Arguments
    ///
    /// * `similarity` - Similarity value between 0.0 and 1.0
    fn similarity_boost(self, similarity: f32) -> Self;

    /// Terminal method that executes speech conversion with a matcher closure.
    ///
    /// This method terminates the fluent chain and executes the speech-to-speech
    /// conversion. The matcher closure receives either the session object on success
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
    /// let audio = FluentVoice::speech_to_speech()
    ///     .from_audio("input.wav")
    ///     .target_voice(VoiceId::new("voice_123"))
    ///     .preserve_emotion(true)
    ///     .convert(|session| {
    ///         Ok => session.into_stream(),
    ///         Err(e) => Err(e),
    ///     })
    ///     .await?;
    /// ```
    fn convert<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Session, VoiceError>) -> R + Send + 'static;
}

/// Static entry point for speech-to-speech conversion.
///
/// This trait provides the static method for starting speech-to-speech
/// conversion operations. Engine implementations typically implement
/// this on a marker struct or their main engine type.
///
/// # Examples
///
/// ```ignore
/// use fluent_voice::prelude::*;
///
/// let conversion = MyEngine::speech_to_speech();
/// ```
pub trait SpeechToSpeechExt {
    /// Begin a new speech-to-speech conversion builder.
    ///
    /// # Returns
    ///
    /// A new speech-to-speech builder instance.
    fn builder() -> impl SpeechToSpeechBuilder;
}