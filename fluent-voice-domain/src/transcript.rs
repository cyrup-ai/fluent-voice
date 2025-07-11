//! Stream & segment traits.
use crate::voice_error::VoiceError;

#[cfg(feature = "async")]
use futures_core::Stream;

/// Individual transcript segment representing a word, phrase, or speech event.
///
/// This trait defines the interface for transcript segments returned by
/// STT engines. Each segment represents a piece of recognized speech
/// with timing and optional speaker information.
pub trait TranscriptSegment {
    /// Start time of this segment in milliseconds from audio start.
    fn start_ms(&self) -> u32;

    /// End time of this segment in milliseconds from audio start.
    fn end_ms(&self) -> u32;

    /// The recognized text content of this segment.
    fn text(&self) -> &str;

    /// Optional speaker identifier for multi-speaker scenarios.
    ///
    /// Returns `Some(speaker_id)` if speaker diarization is enabled
    /// and a speaker was identified, `None` otherwise.
    fn speaker_id(&self) -> Option<&str>;
}

/// Stream of transcript segments from an STT engine.
///
/// This trait represents an async stream that yields transcript segments
/// as they become available from the speech recognition engine. Each
/// item in the stream is a `Result` containing either a segment or an error.
#[cfg(feature = "async")]
pub trait TranscriptStream:
    Stream<Item = Result<Self::Segment, VoiceError>> + Send + Unpin
{
    /// The type of transcript segment yielded by this stream.
    type Segment: TranscriptSegment;
}

/// Blanket implementation for any stream that yields transcript segments.
#[cfg(feature = "async")]
impl<T, S> TranscriptStream for T
where
    T: Stream<Item = Result<S, VoiceError>> + Send + Unpin,
    S: TranscriptSegment,
{
    type Segment = S;
}
