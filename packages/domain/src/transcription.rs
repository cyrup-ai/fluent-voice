//! Stream & segment traits.
use crate::voice_error::VoiceError;
use futures_core::Stream;

/// Individual transcript segment representing a word, phrase, or speech event.
///
/// This trait defines the interface for transcript segments returned by
/// STT engines. Each segment represents a piece of recognized speech
/// with timing and optional speaker information.
pub trait TranscriptionSegment {
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
pub trait TranscriptionStream:
    Stream<Item = Result<Self::Segment, VoiceError>> + Send + Unpin
{
    /// The type of transcript segment yielded by this stream.
    type Segment: TranscriptionSegment;
}

/// Blanket implementation for any stream that yields transcript segments.
impl<T, S> TranscriptionStream for T
where
    T: Stream<Item = Result<S, VoiceError>> + Send + Unpin,
    S: TranscriptionSegment,
{
    type Segment = S;
}

/// Production-quality, zero-allocation transcript segment implementation.
///
/// Uses `Cow<str>` for zero-copy string operations when possible,
/// optimized for high-performance real-time transcription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionSegmentImpl {
    /// Recognized text content (zero-copy when possible)
    text: std::borrow::Cow<'static, str>,
    /// Start time in milliseconds (u32 for efficiency)
    start_ms: u32,
    /// End time in milliseconds (u32 for efficiency)
    end_ms: u32,
    /// Optional speaker identifier (zero-copy when possible)
    speaker_id: Option<std::borrow::Cow<'static, str>>,
}

impl TranscriptionSegmentImpl {
    /// Create a new transcript segment with owned strings.
    #[inline]
    pub fn new(text: String, start_ms: u32, end_ms: u32, speaker_id: Option<String>) -> Self {
        Self {
            text: std::borrow::Cow::Owned(text),
            start_ms,
            end_ms,
            speaker_id: speaker_id.map(std::borrow::Cow::Owned),
        }
    }

    /// Create a new transcript segment with borrowed strings (zero-copy).
    #[inline]
    pub const fn new_borrowed(
        text: &'static str,
        start_ms: u32,
        end_ms: u32,
        speaker_id: Option<&'static str>,
    ) -> Self {
        Self {
            text: std::borrow::Cow::Borrowed(text),
            start_ms,
            end_ms,
            speaker_id: match speaker_id {
                Some(id) => Some(std::borrow::Cow::Borrowed(id)),
                None => None,
            },
        }
    }

    /// Create an empty segment (useful for error recovery).
    #[inline]
    pub const fn empty() -> Self {
        Self {
            text: std::borrow::Cow::Borrowed(""),
            start_ms: 0,
            end_ms: 0,
            speaker_id: None,
        }
    }

    /// Get the duration of this segment in milliseconds.
    #[inline]
    pub const fn duration_ms(&self) -> u32 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Check if this segment contains actual speech content.
    #[inline]
    pub fn has_content(&self) -> bool {
        !self.text.trim().is_empty()
    }
}

impl TranscriptionSegment for TranscriptionSegmentImpl {
    #[inline]
    fn start_ms(&self) -> u32 {
        self.start_ms
    }

    #[inline]
    fn end_ms(&self) -> u32 {
        self.end_ms
    }

    #[inline]
    fn text(&self) -> &str {
        &self.text
    }

    #[inline]
    fn speaker_id(&self) -> Option<&str> {
        self.speaker_id.as_deref()
    }
}
