//! Audio chunk types for TTS streaming
//!
//! This module defines the structured audio chunk types that are used
//! in TTS streaming operations, providing rich metadata and audio data.

use crate::voice_error::VoiceError;

/// Represents a chunk of synthesized audio with metadata
///
/// This type provides structured access to audio data along with
/// timing information, speaker details, and other metadata that
/// enables rich streaming experiences.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// The raw audio data as bytes
    pub data: Vec<u8>,
    /// Duration of this chunk in milliseconds
    pub duration_ms: u64,
    /// Cumulative start time in milliseconds
    pub start_ms: u64,
    /// Speaker identifier for this chunk
    pub speaker_id: Option<String>,
    /// Text that generated this audio chunk
    pub text: Option<String>,
    /// Audio format metadata
    pub format: Option<crate::audio_format::AudioFormat>,
}

impl AudioChunk {
    /// Create a new audio chunk with the given data
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            duration_ms: 0,
            start_ms: 0,
            speaker_id: None,
            text: None,
            format: None,
        }
    }

    /// Create a new audio chunk with full metadata
    pub fn with_metadata(
        data: Vec<u8>,
        duration_ms: u64,
        start_ms: u64,
        speaker_id: Option<String>,
        text: Option<String>,
        format: Option<crate::audio_format::AudioFormat>,
    ) -> Self {
        Self {
            data,
            duration_ms,
            start_ms,
            speaker_id,
            text,
            format,
        }
    }

    /// Get the audio data as bytes
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get the duration in milliseconds
    pub fn duration_ms(&self) -> u64 {
        self.duration_ms
    }

    /// Get the start time in milliseconds
    pub fn start_ms(&self) -> u64 {
        self.start_ms
    }

    /// Get the speaker identifier
    pub fn speaker_id(&self) -> Option<&str> {
        self.speaker_id.as_deref()
    }

    /// Get the source text
    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    /// Get the audio format
    pub fn format(&self) -> Option<&crate::audio_format::AudioFormat> {
        self.format.as_ref()
    }

    /// Convert this audio chunk into raw bytes
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }

    /// Get the size of the audio data in bytes
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the audio chunk is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl From<Vec<u8>> for AudioChunk {
    fn from(data: Vec<u8>) -> Self {
        Self::new(data)
    }
}

impl From<AudioChunk> for Vec<u8> {
    fn from(chunk: AudioChunk) -> Self {
        chunk.data
    }
}

/// Result type for audio chunk operations
pub type AudioChunkResult = Result<AudioChunk, VoiceError>;
