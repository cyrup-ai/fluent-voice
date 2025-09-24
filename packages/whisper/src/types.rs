//! Common data structures shared by the `whisper` sub-module.
//
//  ┌──────────────────────────────────────────────────────────────┐
//  │                       IMPORTANT NOTE                        │
//  │                                                              │
//  │  Down-stream crates consume an `AsyncStream<TtsChunk>`.      │
//  │  Nothing is executed until the first poll, and callers can   │
//  │  map to pure `String` via `.as_text()` or any other helper.  │
//  └──────────────────────────────────────────────────────────────┘

use fluent_voice_domain::TranscriptionSegment;

/// One chunk of transcribed speech produced by the Whisper decoder.
///
/// This flattens the internal `Segment` + `DecodingResult` pair into a
/// single value that is easy to ship across an `mpsc` channel or
/// expose as a stream item.
///
/// Marked **`#[non_exhaustive]`** so the struct can grow (word-level
/// timing, speaker ID, etc.) without breaking SemVer.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TtsChunk {
    /* ---------- timing ---------- */
    /// Absolute start time (seconds since the beginning of the recording).
    pub start: f64,
    /// Absolute end time (seconds).
    pub end: f64,
    /// Cached convenience field (`end - start`).
    pub duration: f64,

    /* ---------- lexical ---------- */
    /// Raw Whisper token IDs.
    pub tokens: Vec<u32>,
    /// Human-readable transcript.
    pub text: String,

    /* ---------- quality metrics ---------- */
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    pub temperature: f64,
    pub compression_ratio: f64,
}

impl TtsChunk {
    /// Construct a new `TtsChunk` from its primitive parts.
    ///
    /// The decoder uses this helper; downstream users rarely need it.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        start: f64,
        end: f64,
        tokens: Vec<u32>,
        text: String,
        avg_logprob: f64,
        no_speech_prob: f64,
        temperature: f64,
        compression_ratio: f64,
    ) -> Self {
        Self {
            start,
            end,
            duration: end - start,
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature,
            compression_ratio,
        }
    }

    /// Create a simple TtsChunk from basic segment information
    /// Used for compatibility with TranscriptionSegment data
    pub fn from_segment(
        text: String,
        start_ms: u32,
        end_ms: u32,
        _speaker_id: Option<String>,
    ) -> Self {
        let start = start_ms as f64 / 1000.0;
        let end = end_ms as f64 / 1000.0;
        Self {
            start,
            end,
            duration: end - start,
            tokens: Vec::new(),
            text,
            avg_logprob: 0.0,
            no_speech_prob: 0.0,
            temperature: 0.0,
            compression_ratio: 0.0,
        }
    }

    /// Sugar: return a `&str` view of the transcript.
    ///
    /// ```ignore
    /// let mut s = stream.map(|c| c.as_text().to_owned());
    /// ```
    pub fn as_text(&self) -> &str {
        &self.text
    }
}

impl TranscriptionSegment for TtsChunk {
    /// Start time of this segment in milliseconds from audio start.
    fn start_ms(&self) -> u32 {
        (self.start * 1000.0) as u32
    }

    /// End time of this segment in milliseconds from audio start.
    fn end_ms(&self) -> u32 {
        (self.end * 1000.0) as u32
    }

    /// The recognized text content of this segment.
    fn text(&self) -> &str {
        &self.text
    }

    /// Optional speaker identifier for multi-speaker scenarios.
    ///
    /// Returns `None` as Whisper doesn't currently support speaker diarization.
    fn speaker_id(&self) -> Option<&str> {
        None
    }
}

/* ----------------------------------------------------------------
Optional internal conversion from the decoder's private `Segment`
type.  Enabled with the crate feature `internal` so this public
module remains decoupled from private implementation details.
---------------------------------------------------------------- */

impl From<crate::whisper::Segment> for TtsChunk {
    fn from(seg: crate::whisper::Segment) -> Self {
        let start = seg.start;
        let end = seg.start + seg.duration;
        let dr = seg.dr;
        Self {
            start,
            end,
            duration: seg.duration,
            tokens: dr.tokens,
            text: dr.text,
            avg_logprob: dr.avg_logprob,
            no_speech_prob: dr.no_speech_prob,
            temperature: dr.temperature,
            compression_ratio: dr.compression_ratio,
        }
    }
}
