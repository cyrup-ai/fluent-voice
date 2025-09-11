//! Direct syntax transformation for examples
//!
//! This module provides concrete implementations that transform the syntax
//! used in examples into working Rust code.

use crate::VoiceError;

/// Trait extension to provide arrow syntax support for TTS builders
pub trait TtsArrowSyntax {
    /// synthesize method that supports arrow syntax
    fn synthesize_arrow<F, R>(self, f: F) -> impl std::future::Future<Output = R> + Send
    where
        F: FnOnce(Result<Self, VoiceError>) -> R + Send + 'static,
        R: Send + 'static,
        Self: Sized;
}

/// Trait extension to provide arrow syntax support for STT builders
pub trait SttArrowSyntax {
    /// listen method that supports arrow syntax
    fn listen_arrow<F, R>(self, f: F) -> impl std::future::Future<Output = R> + Send
    where
        F: FnOnce(Result<Self, VoiceError>) -> R + Send + 'static,
        R: Send + 'static,
        Self: Sized;
}

// Note: The actual macro definitions are in src/macros.rs
// This module provides trait extensions for arrow syntax support

// Re-export the macros from the main macros module
pub use crate::{listen_transform, synthesize_transform};
