//! Default STT Engine Implementation
//!
//! Production-Quality STT Engine: Zero-Allocation, Blazing-Fast, Lock-Free.
//!
//! This module provides the ultimate speech-to-text engine implementation with
//! modular architecture for maintainability and performance.

pub mod audio_processor;
pub mod builders;
pub mod config;
pub mod conversation;
pub mod diagnostics;
pub mod engine;
pub mod stream_impl;
pub mod types;

// Re-export main types for convenience
pub use audio_processor::{AudioProcessor, AudioStream};
pub use builders::{
    DefaultMicrophoneBuilder, DefaultSTTConversationBuilder, DefaultSTTEngineBuilder,
    DefaultSTTPostChunkBuilder, DefaultTranscriptionBuilder,
};
pub use config::{VadConfig, WakeWordConfig};
pub use conversation::DefaultSTTConversation;
pub use engine::DefaultSTTEngine;
pub use types::{
    DefaultTranscriptionSegment, SendableClosure, StreamControl, TranscriptionSegmentWrapper,
};
