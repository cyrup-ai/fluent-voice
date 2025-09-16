//! Synthesis chunk types for streaming TTS operations
//!
//! This module defines the SynthesisChunk wrapper type that enables
//! cyrup_sugars streaming patterns for TTS synthesis operations.

use crate::{AudioChunk, VoiceError};
use cyrup_sugars::prelude::MessageChunk;

/// Synthesis chunk that contains audio data with success/error state
///
/// This type implements the cyrup_sugars pattern for streaming operations
/// while maintaining the NotResult constraint by not containing Result types.
/// It wraps AudioChunk with success/error state for streaming patterns.
#[derive(Debug, Clone)]
pub struct SynthesisChunk {
    /// The audio chunk data
    pub chunk: AudioChunk,
    /// Whether this represents a successful synthesis
    pub success: bool,
    /// Optional error message if synthesis failed
    pub error_message: Option<String>,
}

impl SynthesisChunk {
    /// Create a successful synthesis chunk
    pub fn ok(chunk: AudioChunk) -> Self {
        Self {
            chunk,
            success: true,
            error_message: None,
        }
    }

    /// Create an error synthesis chunk
    pub fn err(error: VoiceError) -> Self {
        Self {
            chunk: AudioChunk::new(Vec::new()),
            success: false,
            error_message: Some(error.to_string()),
        }
    }

    /// Check if this chunk contains a successful audio chunk
    pub fn is_ok(&self) -> bool {
        self.success
    }

    /// Check if this chunk contains an error
    pub fn is_err(&self) -> bool {
        !self.success
    }

    /// Get the audio chunk (always available)
    pub fn chunk(&self) -> &AudioChunk {
        &self.chunk
    }

    /// Get the error message if any
    pub fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    /// Convert into an AudioChunk, following the cyrup_sugars pattern
    /// This enables the `synthesis_chunk.into_chunk()` syntax
    pub fn into_chunk(self) -> AudioChunk {
        self.chunk
    }
}

// Implement From conversions for fluent usage
impl From<AudioChunk> for SynthesisChunk {
    fn from(chunk: AudioChunk) -> Self {
        Self::ok(chunk)
    }
}

impl From<VoiceError> for SynthesisChunk {
    fn from(error: VoiceError) -> Self {
        Self::err(error)
    }
}

impl From<Result<AudioChunk, VoiceError>> for SynthesisChunk {
    fn from(result: Result<AudioChunk, VoiceError>) -> Self {
        match result {
            Ok(chunk) => Self::ok(chunk),
            Err(error) => Self::err(error),
        }
    }
}

// Implement cyrup_sugars NotResult pattern
impl cyrup_sugars::NotResult for SynthesisChunk {}

// Implement MessageChunk trait for SynthesisChunk
impl MessageChunk for SynthesisChunk {
    fn bad_chunk(error: String) -> Self {
        Self {
            chunk: MessageChunk::bad_chunk(error.clone()),
            success: false,
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}
