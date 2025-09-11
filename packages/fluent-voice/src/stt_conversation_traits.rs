//! STT conversation builder traits.

use fluent_voice_domain::{SttConversation, VoiceError, TranscriptionSegment};
use futures_core::Stream;

/// Builder trait for STT conversations.
pub trait SttConversationBuilder: Sized + Send {
    /// The concrete conversation type produced by this builder.
    type Conversation: SttConversation;

    /// Configure for microphone input.
    fn with_microphone(self, device: impl Into<String>) -> impl MicrophoneBuilder;

    /// Configure for file transcription.
    fn transcribe(self, path: impl Into<String>) -> impl TranscriptionBuilder;

    /// Execute recognition and return a transcript stream with JSON syntax support.
    fn listen<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: Stream<Item = TranscriptionSegment> + Send + Unpin + 'static;
}

/// Builder trait for microphone input.
pub trait MicrophoneBuilder: Sized + Send {
    /// The concrete conversation type produced by this builder.
    type Conversation: SttConversation;

    /// Execute recognition and return a transcript stream with JSON syntax support.
    fn listen<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: Stream<Item = TranscriptionSegment> + Send + Unpin + 'static;
}

/// Builder trait for file transcription.
pub trait TranscriptionBuilder: Sized + Send {
    /// The transcript collection type for convenience methods.
    type Transcript: Send;

    /// Execute transcription and return a transcript stream with JSON syntax support.
    ///
    /// This method is IDENTICAL to listen() but for audio files instead of microphone.
    fn transcribe<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Transcript, VoiceError>) -> S + Send + 'static,
        S: Stream<Item = TranscriptionSegment> + Send + Unpin + 'static;
}
