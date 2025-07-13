//! Transcript metadata toggles (aligns with ElevenLabs & OpenAI).
use serde::{Deserialize, Serialize};

/// Granularity level for timestamp information in transcripts.
///
/// Controls how detailed the timing information should be in the
/// transcription output. Different engines may support different
/// levels of granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimestampsGranularity {
    /// No timestamp information included.
    None,
    /// Timestamps at word boundaries.
    Word,
    /// Timestamps at character level (if supported by engine).
    Character,
}

/// Toggle for word-level timestamp inclusion.
///
/// When enabled, each transcribed word will include timing information
/// indicating when it was spoken relative to the audio stream start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WordTimestamps {
    /// Do not include word timestamps.
    Off,
    /// Include timing information for each word.
    On,
}

/// Toggle for speaker diarization in multi-speaker audio.
///
/// When enabled, the transcription will attempt to identify and
/// label different speakers in the audio stream. This is useful
/// for conversations, meetings, and interviews.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Diarization {
    /// Single speaker mode - no speaker labeling.
    Off,
    /// Multi-speaker mode with speaker identification.
    On,
}

/// Toggle for automatic punctuation insertion.
///
/// When enabled, the transcription engine will automatically
/// insert punctuation marks (periods, commas, question marks, etc.)
/// based on speech patterns and pauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Punctuation {
    /// Raw transcription without automatic punctuation.
    Off,
    /// Include automatic punctuation in transcripts.
    On,
}
