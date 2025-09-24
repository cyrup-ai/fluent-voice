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
//!
//! # Optional Feature Dependencies
//!
//! This crate supports additional functionality through optional dependencies:
//!
//! ## Web Server Features
//! - `axum` - High-performance async web server for webhook endpoints
//! - `dashmap` - Concurrent hash map for managing active connections
//!
//! ## Conversational AI Features  
//! - `elevenlabs_convai` - Advanced conversational AI capabilities and voice cloning
//!
//! ## Telephony Integration
//! - `rusty_twilio` - Complete Twilio SDK for phone calls and SMS integration
//!
//! ## Security & Streaming
//! - `secrecy` - Secure handling of API keys and sensitive configuration
//! - `tokio_stream` - Async stream utilities for real-time audio processing
//!
//! ## Cryptographic Utilities
//! - `hex` - Hexadecimal encoding/decoding for webhook signatures
//! - `sha2` - SHA-2 family hash functions for advanced signature validation
//!
//! ## Feature Flags
//!
//! Enable these features in your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! fluent_voice_elevenlabs = { version = "0.1", features = ["web-server", "telephony", "crypto"] }
//! ```
//!
//! ## Feature Groups
//!
//! - **web-server**: Enables `axum`, `dashmap` for webhook servers
//! - **telephony**: Enables `rusty_twilio` for phone integration
//! - **crypto**: Enables `hex`, `sha2` for advanced signature validation
//! - **full**: Enables all optional features

// Internal modules - NOT exposed to users
mod client;
pub mod endpoints;
pub mod error;
pub mod shared;
mod tts;
pub mod utils;

// Timestamp modules - internal implementation
mod timestamp_export;
mod timestamp_metadata;

// Internal engine module - NOT exposed
mod engine;

// Audio format detection, decoding, and encoding modules
pub mod audio_decoders;
pub mod audio_encoders;
pub mod audio_format_detection;

// Twilio integration module - available with telephony features
// pub mod twilio;

// Fluent API module - contains builders but only FluentVoice is exposed
// mod fluent_api; // Removed - replaced with fluent_voice_impl

// FluentVoice trait implementation module
mod fluent_voice_impl;

// Public modules
pub mod voice;

// Re-export timestamp configuration types for domain integration
pub use timestamp_metadata::{TimestampConfiguration, TimestampMetadata};

// Re-export FluentVoice struct implementation
pub use fluent_voice_impl::ElevenLabsFluentVoice; // Direct API that actually works
pub use fluent_voice_impl::FluentVoice;
pub use fluent_voice_impl::Result;

// Re-export fluent-voice macros for arrow syntax support
pub use fluent_voice::{
    arrow_syntax, fv_match, listen_transform, synthesize_transform, tts_synthesize,
};

// Re-export prelude for convenience
pub mod prelude {
    // Import everything from fluent-voice prelude EXCEPT the default FluentVoice
    pub use fluent_voice::prelude::{
        AudioChunk,
        AudioFormat,
        AudioStreamExt, // Add missing AudioStreamExt trait for .play() method
        Diarization,
        Language,
        MicBackend,
        MicrophoneBuilder,
        ModelId,
        Punctuation,
        Similarity,
        Speaker,
        SpeakerBoost,
        SpeakerBuilder, // Add missing SpeakerBuilder trait
        SpeechSource,
        Stability,
        SttConversationBuilder,
        SttPostChunkBuilder,
        StyleExaggeration,
        TimestampsGranularity,
        TranscriptionBuilder,
        TtsConversationBuilder,
        TtsConversationChunkBuilder,
        VadMode,
        VocalSpeedMod,
        VoiceError,
        VoiceId,
        WordTimestamps,
    };

    // Import transcription types from domain
    pub use fluent_voice_domain::transcription::{TranscriptionSegment, TranscriptionSegmentImpl};

    // Import TtsConversation trait for into_stream method
    pub use fluent_voice_domain::TtsConversation;

    // Import SttConversation trait for into_stream method (from fluent_voice, not domain)
    pub use fluent_voice::stt_conversation::SttConversation;

    // Use ElevenLabs FluentVoice implementation instead of default
    pub use super::ElevenLabsFluentVoice;
    pub use super::FluentVoice;
}
