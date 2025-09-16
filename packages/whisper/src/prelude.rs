//! Prelude module for fluent-voice-whisper
//!
//! This module re-exports the most commonly used types and functions from the whisper library.
//! Users can import everything they need with a single use statement:
//!
//! ```rust
//! use fluent_voice_whisper::prelude::*;
//! ```

// Core whisper types
pub use crate::whisper::{Decoder, DecodingResult, Segment, Task, WhichModel};

// Builder API types
pub use crate::builder::{ModelConfig, WhisperTranscriber};

// Streaming and transcript types
pub use crate::stream::WhisperStream;
pub use crate::transcript::Transcript;
pub use crate::types::TtsChunk;

// Audio processing
pub use crate::pcm_decode::pcm_decode;

// Model and token utilities (feature-dependent)
#[cfg(feature = "microphone")]
pub use crate::microphone::{Model, token_id};

#[cfg(not(feature = "microphone"))]
pub use crate::whisper::{Model, token_id};

// Multilingual support
pub use crate::multilingual;
