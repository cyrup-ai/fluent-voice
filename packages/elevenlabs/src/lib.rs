//! ElevenLabs fluent-voice builder API
//!
//! This crate provides ONLY the fluent-voice builder interfaces for ElevenLabs.
//! Internal engine implementation is private - use the builder API.
//!
//! # Usage
//!
//! ```no_run
//! use fluent_voice_elevenlabs::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), VoiceError> {
//!     // TTS using fluent builder API
//!     let audio = FluentVoice::tts()
//!         .api_key_from_env()?
//!         .http3_enabled(true)
//!         .with_speaker(|builder| {
//!             Ok(builder
//!                 .named("Sarah")
//!                 .speak("Hello from fluent-voice!")
//!                 .build())
//!         })?
//!         .synthesize(|result| result)
//!         .await?;
//!
//!     // STT using fluent builder API  
//!     let transcript = FluentVoice::stt()
//!         .api_key_from_env()?
//!         .http3_enabled(true)
//!         .transcribe("audio.wav")?
//!         .with_word_timestamps()
//!         .emit(|result| result)
//!         .await?;
//!
//!     Ok(())
//! }
//! ```

// Internal modules - NOT exposed to users
mod client;
pub mod endpoints;
pub mod error;
pub mod shared;
mod tts;
mod utils;

// Timestamp modules - internal implementation
mod timestamp_export;
mod timestamp_metadata;

// Internal engine module - NOT exposed
mod engine;

// Audio format detection and decoding modules
pub mod audio_decoders;
pub mod audio_format_detection;

// Fluent API module - contains builders but only FluentVoice is exposed
mod fluent_api;

// Public modules
pub mod voice;

// Re-export timestamp configuration types for domain integration
pub use timestamp_metadata::{TimestampConfiguration, TimestampMetadata};

// Re-export ONLY FluentVoice entry point - hide raw engine bypasses
pub use fluent_api::{FluentVoice, Result};
