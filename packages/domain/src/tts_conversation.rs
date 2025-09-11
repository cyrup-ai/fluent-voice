//! TTS conversation domain objects.

use futures_core::Stream;

/// Engine-specific conversation object.
///
/// This trait represents a completed TTS conversation that has been
/// configured and is ready to produce audio output. Engine implementations
/// provide concrete types that implement this trait.
pub trait TtsConversation: Send {
    /// Async audio stream of structured AudioChunk objects.
    ///
    /// The audio stream contains AudioChunk objects with metadata including
    /// timing, speaker information, text, and raw audio data that can be
    /// played through audio output devices like rodio.
    type AudioStream: Stream<Item = crate::AudioChunk> + Send + Unpin;

    /// Convert this conversation into an audio stream.
    ///
    /// This method consumes the conversation and returns the underlying
    /// audio stream that can be used to play or process the synthesized audio.
    fn into_stream(self) -> Self::AudioStream;
}

// All builder traits have been moved to fluent-voice package where they belong
