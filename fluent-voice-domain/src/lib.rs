//! # Fluent Voice Domain
//!
//! Shared domain objects and types for the fluent-voice ecosystem.
//!
//! This crate contains the core domain types that are shared between
//! different components of the fluent-voice system, enabling clean
//! separation of concerns and breaking cyclic dependencies.

pub mod audio_format;
pub mod fluent_voice;
pub mod language;
pub mod mic_backend;
pub mod model_id;
pub mod noise_reduction;
pub mod speaker;
pub mod speaker_builder;
pub mod speech_source;
pub mod stt_conversation;
pub mod stt_engine;
pub mod timestamps;
pub mod transcript;
pub mod tts_conversation;
pub mod tts_engine;
pub mod vad_mode;
pub mod vocal_speed;
pub mod voice_error;
pub mod voice_id;

// Re-export all public types
pub use audio_format::AudioFormat;
pub use fluent_voice::FluentVoice;
pub use language::Language;
pub use mic_backend::MicBackend;
pub use model_id::ModelId;  
pub use noise_reduction::NoiseReduction;
pub use speaker::Speaker;
pub use speaker_builder::{SpeakerBuilder, SpeakerExt};
pub use speech_source::SpeechSource;
pub use stt_conversation::{SttConversation, SttConversationBuilder, MicrophoneBuilder, TranscriptionBuilder};
pub use stt_engine::SttEngine;
pub use timestamps::{Diarization, Punctuation, TimestampsGranularity, WordTimestamps};
pub use transcript::{TranscriptSegment, TranscriptStream};
pub use tts_conversation::{TtsConversation, TtsConversationBuilder, TtsConversationExt};
pub use tts_engine::TtsEngine;
pub use vad_mode::VadMode;
pub use vocal_speed::VocalSpeedMod;
pub use voice_error::VoiceError;
pub use voice_id::VoiceId;

/// Prelude module containing commonly used types.
pub mod prelude {
    pub use crate::{
        AudioFormat, Diarization, Language, MicBackend, ModelId, Punctuation, SpeechSource,
        TimestampsGranularity, TranscriptSegment, VadMode, VocalSpeedMod, VoiceError, VoiceId,
        WordTimestamps,
    };
}
