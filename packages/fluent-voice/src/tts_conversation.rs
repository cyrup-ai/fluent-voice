//! TTS conversation builder traits.

use fluent_voice_domain::{TtsConversation, VoiceError, AudioChunk, Language};
use futures_core::Stream;

/// Builder trait for TTS conversations.
///
/// This trait provides the fluent API for configuring and executing
/// text-to-speech synthesis. All builder methods belong in fluent-voice package.
pub trait TtsConversationBuilder: Sized + Send {
    /// The concrete conversation type produced by this builder.
    type Conversation: TtsConversation;

    /// Execute synthesis and return an audio stream with JSON syntax support.
    ///
    /// This method terminates the fluent chain and executes the TTS synthesis,
    /// returning a stream of audio chunks that can be processed with the fluent
    /// `.play()` method for real-time audio playback.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let stream = FluentVoice::tts()
    ///     .with_speaker(speaker)
    ///     .synthesize(|conversation| {
    ///         Ok => conversation.into_stream(),
    ///         Err(e) => Err(e),
    ///     });
    /// ```
    fn synthesize<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: Stream<Item = AudioChunk> + Send + Unpin + 'static;
}

/// Trait for chunk-by-chunk processing of TTS synthesis.
pub trait TtsConversationChunkBuilder: Sized + Send {
    /// The concrete conversation type produced by this chunk builder.
    type Conversation: TtsConversation;

    /// Terminal method that executes synthesis with chunk processing.
    fn synthesize(self) -> impl Stream<Item = AudioChunk> + Send + Unpin;
}
