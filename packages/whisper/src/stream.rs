//! WhisperStream that implements the fluent-voice TranscriptStream trait.

use futures_core::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::types::TtsChunk;
use fluent_voice_domain::VoiceError;

/// Async stream of Whisper transcript chunks.
///
/// This type implements the fluent-voice `TranscriptStream` trait while
/// maintaining compatibility with the existing Whisper builder API.
pub struct WhisperStream {
    receiver: UnboundedReceiver<TtsChunk>,
}

impl WhisperStream {
    /// Create a new WhisperStream from an unbounded receiver.
    #[allow(dead_code)] // Library code - used by fluent-voice builders
    pub(crate) fn new(receiver: UnboundedReceiver<TtsChunk>) -> Self {
        Self { receiver }
    }
}

impl Stream for WhisperStream {
    type Item = Result<TtsChunk, VoiceError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.receiver.poll_recv(cx) {
            Poll::Ready(Some(chunk)) => Poll::Ready(Some(Ok(chunk))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

// TranscriptStream is implemented via blanket impl for streams that yield TranscriptSegment

// Ensure the stream is Unpin for TranscriptStream requirement
impl Unpin for WhisperStream {}
