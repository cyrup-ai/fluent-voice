//! Concrete STT builder implementation
//!
//! This module provides a non-macro implementation of the STT conversation builders
//! that can be used as a base for engine-specific implementations.

pub mod builder_module;
pub mod chunk_handler;
pub mod config_methods;
pub mod conversation_builder_core;
pub mod conversation_builder_trait;
pub mod conversation_impl;
pub mod microphone_builder;
pub mod microphone_builder_trait;
pub mod post_chunk_builder;
pub mod post_chunk_builder_trait;
pub mod transcript_impl;
pub mod transcription_builder;
pub mod transcription_builder_trait;
pub mod transcription_segment_wrapper;

// Re-export main types for backward compatibility
pub use conversation_builder_core::SttConversationBuilderImpl;
pub use conversation_impl::SttConversationImpl;
pub use microphone_builder::MicrophoneBuilderImpl;
pub use transcript_impl::TranscriptImpl;
pub use transcription_builder::TranscriptionBuilderImpl;

// Re-export builder module
pub mod builder {
    pub use super::builder_module::*;
}
