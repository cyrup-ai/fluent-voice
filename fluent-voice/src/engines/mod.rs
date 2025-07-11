//! Engine trait implementations for FluentVoice.
//!
//! This module contains trait implementations that extend
//! the core FluentVoice functionality.

mod default_stt_engine;

pub use default_stt_engine::{
    DefaultSTTEngine, DefaultSTTEngineBuilder, VadConfig, WakeWordConfig,
};
