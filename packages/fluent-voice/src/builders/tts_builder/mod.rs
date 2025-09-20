//! Zero-allocation TTS builder implementation with arrow syntax support
//!
//! This module provides a blazing-fast, zero-allocation TTS conversation builder
//! with full arrow syntax support through cyrup_sugars integration.

pub mod audio_chunk_wrapper;
pub mod audio_processing;
pub mod builder_core;
pub mod builder_module;
pub mod chunk_builder_trait;
pub mod chunk_handler;
pub mod chunk_synthesis;
pub mod config_methods;
pub mod conversation_builder_trait;
pub mod conversation_impl;
pub mod speaker_line;
pub mod speaker_line_builder;
pub mod speaker_processing;
pub mod synthesis;

// Re-export main types for backward compatibility
pub use builder_core::TtsConversationBuilderImpl;
pub use conversation_impl::TtsConversationImpl;
pub use speaker_line::SpeakerLine;
pub use speaker_line_builder::SpeakerLineBuilder;

// Re-export builder module
pub mod builder {
    pub use super::builder_module::*;
}
