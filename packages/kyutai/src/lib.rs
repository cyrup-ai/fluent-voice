//! Moshi Language Model Implementation for Fluent Voice
//!
//! This crate provides a Rust implementation of the Moshi Language Model System
//!
//! A comprehensive language model system built with the Candle framework,
//! providing state-of-the-art text generation capabilities with conditioning support.

pub mod asr;
pub mod audio;
pub mod conditioner;
pub mod config;
pub mod conv;
pub mod engine;
pub mod error;
pub mod generator;
pub mod lm;
pub mod lm_generate;
pub mod mimi;
pub mod model;
pub mod models;
pub mod nn;
pub mod quantization;
pub mod sampling_config;
// pub mod seanet; // TODO: implement missing module
pub mod speech_generator;
// pub mod stream_both; // TODO: implement missing module
pub mod streaming;
pub mod tokenizer;
pub mod transformer;
pub mod tts;
pub mod tts_streaming;
pub mod utils;
pub mod visualizer;
pub mod wav;

// Re-export essential types and modules for ease of use
pub use self::asr::{State as AsrState, Word};
pub use self::engine::{KyutaiEngine, KyutaiSttConversationBuilder, KyutaiTtsConversationBuilder};
pub use self::error::{MoshiError, Result};
pub use self::models::{
    KyutaiModelConfig, KyutaiModelManager, KyutaiModelPaths, download_kyutai_models,
    download_kyutai_models_with_config,
};
pub use self::speech_generator::{
    AudioStream, GenerationStats, SpeechGenerationError, SpeechGenerator, SpeechGeneratorBuilder,
    VoiceParameters,
};

// TODO: Uncomment when modules are implemented
// pub use self::lm::LmModel;
pub use self::mimi::Mimi;
// pub use self::stream_both::{
//     Config as StreamConfig, LiveKitAudioIntegration, LiveKitAudioReceiver, StreamOut,
//     StreamingModel,
// };
// pub use self::tts::Config as TtsModelConfig;
// pub use self::tts::Model as TtsModel;
// pub use livekit::prelude::{Room, RoomOptions};

// Public re-exports of underlying libraries
extern crate candle_core as candle;
pub use candle_nn;

// Constants
pub const DEFAULT_SAMPLE_RATE: usize = 24_000;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default model configurations
pub mod defaults {
    use crate::config::Config;

    /// Small model configuration (suitable for testing)
    pub fn small_config() -> Config {
        Config::default()
    }

    /// Base model configuration
    pub fn base_config() -> Config {
        Config::default()
    }

    /// Large model configuration
    pub fn large_config() -> Config {
        Config::default()
    }
}

// Note: Result is already re-exported from error module, no need to redefine
