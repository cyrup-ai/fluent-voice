use crate::AudioChunk;
use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Zero-allocation, blazing-fast audio stream wrapper with fluent `.play()` method
///
/// Encapsulates streaming audio chunks and provides elegant ergonomic audio playback.
/// This is a domain type that can be implemented by different audio backends.
pub struct AudioStream {
    stream: Pin<Box<dyn Stream<Item = AudioChunk> + Send + Unpin>>,
}

impl AudioStream {
    /// Create a new AudioStream wrapper with zero allocation
    #[inline]
    pub fn new(stream: Pin<Box<dyn Stream<Item = AudioChunk> + Send + Unpin>>) -> Self {
        Self { stream }
    }

    /// Create AudioStream from any compatible stream
    #[inline]
    pub fn from_stream<S>(stream: S) -> Self
    where
        S: Stream<Item = AudioChunk> + Send + Unpin + 'static,
    {
        Self {
            stream: Box::pin(stream),
        }
    }
}

impl Stream for AudioStream {
    type Item = AudioChunk;

    #[inline]
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}

// Note: The .play() method will be implemented by the concrete backend (fluent-voice crate)
// through an extension trait to avoid circular dependencies with rodio
