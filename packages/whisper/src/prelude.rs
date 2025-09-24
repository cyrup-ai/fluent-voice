//! Prelude module for fluent-voice-whisper
//!
//! This module re-exports only the essential public API for users.
//! Import everything you need with:
//!
//! ```rust
//! use fluent_voice_whisper::prelude::*;
//! ```

// Public fluent builder API - The main interface
pub use crate::builder::{WhisperConversation, WhisperSttBuilder};

// Essential domain types from fluent-voice-domain
pub use fluent_voice_domain::prelude::*;
