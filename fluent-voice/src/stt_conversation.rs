//! Live/batch transcription builder.
use crate::{
    language::Language,
    noise_reduction::NoiseReduction,
    speech_source::SpeechSource,
    timestamps::{Diarization, Punctuation, TimestampsGranularity, WordTimestamps},
    transcript::TranscriptStream,
    vad_mode::VadMode,
    voice_error::VoiceError,
};
use core::future::Future;
use futures_core::Stream;

/// Engine-specific STT session object.
///
/// This trait represents a configured speech-to-text session that is
/// ready to produce a transcript stream. Engine implementations provide
/// concrete types that implement this trait.
pub trait SttConversation: Send {
    /// The transcript stream type that will be produced.
    type Stream: TranscriptStream;

    /// Convert this session into a transcript stream.
    ///
    /// This method consumes the session and returns the underlying
    /// transcript stream that yields recognition results.
    fn into_stream(self) -> Self::Stream;
}

/// Fluent builder for STT conversations.
///
/// This trait provides the builder interface for configuring speech-to-text
/// sessions with audio sources, language hints, VAD settings, and other
/// recognition parameters.
pub trait SttConversationBuilder: Sized + Send {
    /* fluent setters */

    /// Specify the audio input source.
    ///
    /// This can be either a file path or live microphone input with
    /// the associated audio format and capture parameters.
    fn with_source(self, src: SpeechSource) -> Self;

    /// Configure voice activity detection mode.
    ///
    /// Controls how aggressively the engine detects speech boundaries
    /// and determines when a speaker has finished talking.
    fn vad_mode(self, mode: VadMode) -> Self;

    /// Set noise reduction level.
    ///
    /// Controls how aggressively background noise is filtered out
    /// before speech recognition processing.
    fn noise_reduction(self, level: NoiseReduction) -> Self;

    /// Provide a language hint for improved accuracy.
    ///
    /// This helps the recognition engine optimize for the expected
    /// language, improving transcription quality.
    fn language_hint(self, lang: Language) -> Self;

    /// Enable or disable speaker diarization.
    ///
    /// When enabled, the engine will attempt to identify and label
    /// different speakers in multi-speaker audio.
    fn diarization(self, d: Diarization) -> Self;

    /// Control word-level timestamp inclusion.
    ///
    /// When enabled, each transcribed word will include timing
    /// information relative to the audio stream.
    fn word_timestamps(self, w: WordTimestamps) -> Self;

    /// Set timestamp granularity level.
    ///
    /// Controls the level of timing detail included in the transcript,
    /// from no timestamps to character-level timing.
    fn timestamps_granularity(self, g: TimestampsGranularity) -> Self;

    /// Enable or disable automatic punctuation insertion.
    ///
    /// When enabled, the engine will automatically add punctuation
    /// based on speech patterns and pauses.
    fn punctuation(self, p: Punctuation) -> Self;

    /* polymorphic branching */

    /// Configure for microphone input.
    fn with_microphone(self, device: impl Into<String>) -> Self;

    /// Configure for file transcription.
    fn transcribe(self, path: impl Into<String>) -> Self;

    /// Attach a progress message template.
    fn with_progress<S: Into<String>>(self, template: S) -> Self;

    /* terminal */

    /// Execute recognition with a matcher closure.
    ///
    /// This method terminates the fluent chain and starts speech recognition.
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
    /// let stream = FluentVoice::stt()
    ///     .with_microphone("default")
    ///     .listen(|conversation| {
    ///         Ok => conversation.into_stream(),
    ///         Err(e) => Err(e),
    ///     })
    ///     .await?;
    /// ```
    fn listen<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Conversation, VoiceError>) -> R + Send + 'static;

    /// Emit a transcript with a matcher closure.
    ///
    /// This method terminates the fluent chain and produces a transcript.
    /// The matcher closure receives either the transcript object on success
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
    /// let stream = FluentVoice::stt()
    ///     .transcribe("audio.wav")
    ///     .emit(|transcript| {
    ///         Ok => transcript.into_stream(),
    ///         Err(e) => Err(e),
    ///     })
    ///     .await?;
    /// ```
    fn emit<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Transcript, VoiceError>) -> R + Send + 'static;

    /// Drain the stream and gather into a complete transcript.
    fn collect(self) -> impl Future<Output = Result<Self::Transcript, VoiceError>> + Send;

    /// Variant that accepts a user-supplied closure to post-process the result.
    fn collect_with<F, R>(self, handler: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Transcript, VoiceError>) -> R + Send + 'static;

    /// Convenience: obtain a stream of plain text segments.
    fn as_text(self) -> impl Stream<Item = String> + Send;

    /// The concrete conversation type produced by this builder.
    type Conversation: SttConversation;

    /// The transcript collection type for convenience methods.
    type Transcript: Send;
}

/// Static entry point for STT conversations.
///
/// This trait provides the static method for starting a new STT conversation.
/// Engine implementations typically implement this on a marker struct or
/// their main engine type.
///
/// # Examples
///
/// ```ignore
/// use fluent_voice::prelude::*;
///
/// let conversation = MyEngine::stt();
/// ```
pub trait SttConversationExt {
    /// Begin a new STT conversation builder.
    ///
    /// # Returns
    ///
    /// A new conversation builder instance.
    fn builder() -> impl SttConversationBuilder;
}
