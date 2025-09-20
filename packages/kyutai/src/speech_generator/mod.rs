//! High-Performance Speech Generation Engine
//!
//! This module provides a blazing-fast, zero-allocation speech generation system
//! with real-time streaming capabilities and comprehensive error handling.

pub mod audio_buffer;
pub mod audio_stream;
pub mod builder;
pub mod config;
pub mod core_generator;
pub mod error;
pub mod speaker_processing;
pub mod stats;
pub mod utils;
pub mod voice_params;
pub mod voice_processing;

// Re-export main types for public API
pub use audio_buffer::AudioBuffer;
pub use audio_stream::{AudioStream, AudioStreamIterator};
pub use builder::{SpeechGeneratorBuilder, convenience};
pub use config::GeneratorConfig;
pub use core_generator::SpeechGenerator;
pub use error::SpeechGenerationError;
pub use stats::GenerationStats;
pub use voice_params::{SpeakerPcmConfig, SpeakerPcmData, VoiceParameters};

// Re-export convenience functions at module level
pub use builder::convenience::{generate_from_models, generate_speech, generate_speech_with_voice};
