//! Engine trait implementations for FluentVoice.
//!
//! This module contains trait implementations that extend
//! the core FluentVoice functionality.

mod default_stt_engine;

pub use default_stt_engine::{
    AudioProcessor, AudioStream, DefaultSTTConversationBuilder, DefaultSTTEngine,
    DefaultSTTEngineBuilder, DefaultSTTPostChunkBuilder, VadConfig, WakeWordConfig,
    WakeWordDetection,
};
