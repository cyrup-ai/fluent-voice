//! # Fluent Voice Domain
//!
//! Shared domain objects and types for the fluent-voice ecosystem.
//! 
//! This crate contains the core domain types that are shared between
//! different components of the fluent-voice system, enabling clean
//! separation of concerns and breaking cyclic dependencies.

pub mod audio_format;
pub mod language;
pub mod mic_backend;
pub mod model_id;
pub mod speech_source;
pub mod timestamps;
pub mod transcript;  
pub mod vad_mode;
pub mod vocal_speed;
pub mod voice_error;
pub mod voice_id;

// Re-export all public types
pub use audio_format::AudioFormat;
pub use language::Language;
pub use mic_backend::MicBackend;
pub use model_id::ModelId;
pub use speech_source::SpeechSource;
pub use timestamps::{TimestampsGranularity, WordTimestamps, Diarization, Punctuation};
pub use transcript::{TranscriptSegment, TranscriptStream};
pub use vad_mode::VadMode;
pub use vocal_speed::VocalSpeedMod;
pub use voice_error::VoiceError;
pub use voice_id::VoiceId;

/// Prelude module containing commonly used types.
pub mod prelude {
    pub use crate::{
        AudioFormat,
        Language,
        MicBackend,
        ModelId,
        SpeechSource,
        TimestampsGranularity,
        WordTimestamps,
        Diarization,
        Punctuation,
        TranscriptSegment,
        VadMode,
        VocalSpeedMod,
        VoiceError,
        VoiceId,
    };
    
    #[cfg(feature = "async")]
    pub use crate::TranscriptStream;
}
