/// A module that provides a websocket client for interacting with an ElevenLabs' Conversational AI Agent.
pub mod client;
/// A module that provides an error type for the ElevenLabs' Conversational AI.
pub mod error;
pub mod messages;

/// An error type for the ElevenLabs Conversational AI.
pub type Result<T> = std::result::Result<T, error::ConvAIError>;

pub use speakrs_elevenlabs::endpoints::convai::*;
pub use speakrs_elevenlabs::{DefaultVoice, ElevenLabsClient, LegacyVoice};
