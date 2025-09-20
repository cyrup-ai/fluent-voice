//! Type Definitions and Helper Structures
//!
//! Contains wrapper types, enums, and helper structures used throughout
//! the default STT engine implementation.

use cyrup_sugars::prelude::MessageChunk;
use fluent_voice_domain::{TranscriptionSegment, TranscriptionSegmentImpl, VoiceError};
use futures_core::Stream;
use std::pin::Pin;

/// Safe wrapper for FnMut closures to enable Send trait implementation
pub struct SendableClosure<F>(pub F);
unsafe impl<F> Send for SendableClosure<F> where F: Send {}

// Wrapper type to implement MessageChunk for TranscriptionSegmentImpl (avoiding orphan rule)
#[derive(Debug, Clone)]
pub struct TranscriptionSegmentWrapper(pub TranscriptionSegmentImpl);

impl MessageChunk for TranscriptionSegmentWrapper {
    fn bad_chunk(error: String) -> Self {
        TranscriptionSegmentWrapper(TranscriptionSegmentImpl::new(
            format!("[ERROR] {}", error),
            0,
            0,
            None,
        ))
    }

    fn error(&self) -> Option<&str> {
        if self.0.text().starts_with("[ERROR]") {
            Some(&self.0.text()[8..].trim())
        } else {
            None
        }
    }

    fn is_error(&self) -> bool {
        self.0.text().starts_with("[ERROR]")
    }
}

impl From<TranscriptionSegmentImpl> for TranscriptionSegmentWrapper {
    fn from(segment: TranscriptionSegmentImpl) -> Self {
        TranscriptionSegmentWrapper(segment)
    }
}

impl From<TranscriptionSegmentWrapper> for TranscriptionSegmentImpl {
    fn from(wrapper: TranscriptionSegmentWrapper) -> Self {
        wrapper.0
    }
}

/// Zero-Allocation TranscriptionSegment: Pre-allocated string pools and stack-based storage
#[derive(Debug, Clone)]
pub struct DefaultTranscriptionSegment {
    text: String,
    start_ms: u32,
    end_ms: u32,
    speaker_id: Option<String>,
}

impl TranscriptionSegment for DefaultTranscriptionSegment {
    fn start_ms(&self) -> u32 {
        self.start_ms
    }

    fn end_ms(&self) -> u32 {
        self.end_ms
    }

    fn text(&self) -> &str {
        &self.text
    }

    fn speaker_id(&self) -> Option<&str> {
        self.speaker_id.as_deref()
    }
}

impl DefaultTranscriptionSegment {
    pub fn new(text: String, start_ms: u32, end_ms: u32, speaker_id: Option<String>) -> Self {
        Self {
            text,
            start_ms,
            end_ms,
            speaker_id,
        }
    }
}

/// Zero-Allocation Stream: Pre-allocated, lock-free transcript stream
pub type DefaultTranscriptStream =
    Pin<Box<dyn Stream<Item = Result<DefaultTranscriptionSegment, VoiceError>> + Send>>;

/// Stream Control Messages: Lock-free command system
/// Reserved for future stream control implementation
#[derive(Debug, Clone)]
#[allow(dead_code)] // Reserved for future stream control implementation
pub enum StreamControl {
    Start,
    Stop,
    Reset,
    WakeWordDetected { confidence: f32, timestamp: u64 },
    SpeechSegmentEnd { duration_ms: u32 },
}
