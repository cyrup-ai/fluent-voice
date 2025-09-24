// Core implementation modules - all internal
mod builder;
mod multilingual;
mod pcm_decode;
mod stream;
mod token_filtering;
mod transcript;
mod types;
mod whisper;

// Microphone feature - conditionally internal
#[cfg(feature = "microphone")]
mod microphone;

// Public prelude with essential API only
pub mod prelude;

// Public fluent API - Primary interface for new code
pub use builder::{WhisperConversation, WhisperSttBuilder};

// Configuration for advanced users
pub use builder::ModelConfig;

// Legacy types for existing integrations
pub use transcript::Transcript;
pub use types::TtsChunk;

// Core whisper functionality for advanced users
pub use whisper::{Decoder, Task, WhichModel, token_id};

// Audio processing utilities
pub use pcm_decode::pcm_decode;

// Conditionally expose Model based on features
#[cfg(feature = "microphone")]
pub use microphone::Model;
#[cfg(not(feature = "microphone"))]
pub use whisper::Model;

// Re-export essential domain types for interoperability
pub use fluent_voice_domain::prelude::*;
