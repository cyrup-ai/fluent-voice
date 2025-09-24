//! # Fluent Voice Domain
//!
//! Shared domain objects and types for the fluent-voice ecosystem.
//!
//! This crate contains the core domain types that are shared between
//! different components of the fluent-voice system, enabling clean
//! separation of concerns and breaking cyclic dependencies.

pub mod audio_chunk;
pub mod audio_format;
pub mod audio_isolation;
pub mod audio_stream;

pub mod fluent_voice;
pub mod language;
pub mod mic_backend;
pub mod model_id;
pub mod noise_reduction;
pub mod pitch_range;
pub mod pronunciation_dict;
pub mod similarity;
pub mod sound_effects;
pub mod speaker;
pub mod speaker_boost;
pub mod speaker_builder;
pub mod speech_source;
pub mod speech_to_speech;
pub mod stability;
pub mod stt_conversation;
pub mod style_exaggeration;
pub mod synthesis_chunk;
pub mod timestamps;
pub mod transcription;
pub mod tts_conversation;
pub mod tts_engine;
pub mod vad_mode;
pub mod vocal_speed;
pub mod voice_clone;
pub mod voice_discovery;
pub mod voice_error;
pub mod voice_id;
pub mod voice_labels;
pub mod wake_word;
pub mod wake_word_conversation;

// Re-export core types
pub use audio_chunk::{AudioChunk, AudioChunkResult};
pub use audio_format::AudioFormat;
pub use audio_stream::AudioStream;
// FluentVoice moved to fluent-voice package
pub use language::Language;
pub use mic_backend::MicBackend;
pub use model_id::ModelId;
pub use noise_reduction::NoiseReduction;
pub use speaker::Speaker;
// SpeakerBuilder moved to fluent-voice package
pub use speech_source::SpeechSource;
pub use stt_conversation::{SttConfig, SttConversation, SttConversationImpl};
pub use timestamps::{
    AudioChunkTimestamp, CharacterTimestamp, Diarization, Punctuation, SynthesisMetadata,
    TimestampConfiguration, TimestampMetadata, TimestampsGranularity, WordTimestamp,
    WordTimestamps,
};
pub use transcription::{TranscriptionSegment, TranscriptionSegmentImpl, TranscriptionStream};
pub use tts_conversation::TtsConversation;
// All builder traits moved to fluent-voice package
// TtsEngine moved to fluent-voice package
pub use synthesis_chunk::SynthesisChunk;
pub use vad_mode::VadMode;
pub use vocal_speed::VocalSpeedMod;
pub use voice_error::VoiceError;
pub use voice_id::VoiceId;
pub use wake_word::{WakeWordDetectionResult, WakeWordEvent};

// Re-export value types only (builder traits moved to fluent-voice package)
pub use pitch_range::*;
pub use pronunciation_dict::*;
pub use similarity::*;
pub use speaker_boost::*;
pub use stability::*;
pub use style_exaggeration::*;
pub use voice_labels::*;
// audio_isolation, sound_effects, speech_to_speech, voice_clone, voice_discovery, wake_word now contain only comments
// wake_word_conversation moved to fluent-voice package

/// Prelude module containing commonly used types.
pub mod prelude {
    pub use crate::{
        AudioChunk, AudioChunkResult, AudioChunkTimestamp, AudioFormat, CharacterTimestamp,
        Diarization, Language, MicBackend, ModelId, Punctuation, SpeechSource, SynthesisChunk,
        SynthesisMetadata, TimestampConfiguration, TimestampMetadata, TimestampsGranularity,
        TranscriptionSegment, TranscriptionSegmentImpl, VadMode, VocalSpeedMod, VoiceError,
        VoiceId, WordTimestamp, WordTimestamps,
    };
}
