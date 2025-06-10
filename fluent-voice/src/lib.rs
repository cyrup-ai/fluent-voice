//! # Fluent Voice API
//!
//! Pure-trait fluent builder API for TTS & STT engines.
//!
//! This crate provides trait-based interfaces for Text-to-Speech (TTS) and
//! Speech-to-Text (STT) engines with a fluent builder pattern that maintains
//! exactly one `.await?` per chain.
//!
//! ## Usage Pattern
//!
//! ### TTS (Text-to-Speech)
//!
//! ```ignore
//! let audio = FluentVoice::conversation()
//!     .with_speaker(
//!         Speaker::named("Bob")
//!             .with_speed_modifier(VocalSpeedMod(0.9))
//!             .speak("Hello, world!")
//!             .build()
//!     )
//!     .synthesize(|conversation| {
//!         Ok => conversation.into_stream(),
//!         Err(e) => Err(e),
//!     })
//!     .await?;
//! ```
//!
//! ### STT (Speech-to-Text)
//!
//! ```ignore
//! let mut segments = Stt::conversation()
//!     .with_source(SpeechSource::Microphone {
//!         backend: MicBackend::Default,
//!         format: AudioFormat::Pcm16Khz,
//!         sample_rate: 16_000,
//!     })
//!     .vad_mode(VadMode::Accurate)
//!     .listen(|conversation| {
//!         Ok => conversation.into_stream(),
//!         Err(e) => Err(e),
//!     })
//!     .await?;
//! ```

/* ───── shared fundamentals ───── */
pub mod audio_format;
pub mod language;
pub mod voice_error;

/* ───── TTS chain ───── */
pub mod model_id;
pub mod pitch_range;
pub mod speaker;
pub mod speaker_builder;
pub mod tts_conversation;
pub mod tts_engine;
pub mod tts_settings;
pub mod vocal_speed;
pub mod voice_id;

/* ───── STT chain ───── */
pub mod mic_backend;
pub mod noise_reduction;
pub mod speech_source;
pub mod stt_conversation;
pub mod stt_engine;
pub mod timestamps;
pub mod transcript;
pub mod vad_mode;

/* ───── internal matcher macro ───── */
mod macros;

/* ───── prelude for users ───── */
pub mod prelude {
    //! Re-exports of commonly used types and traits.

    /* shared */
    pub use crate::{audio_format::AudioFormat, language::Language, voice_error::VoiceError};

    /* TTS */
    pub use crate::{
        model_id::ModelId,
        pitch_range::PitchRange,
        speaker::Speaker,
        speaker_builder::{SpeakerBuilder, SpeakerExt},
        tts_conversation::{TtsConversationBuilder, TtsConversationExt},
        tts_engine::TtsEngine,
        tts_settings::{Similarity, SpeakerBoost, Stability, StyleExaggeration},
        vocal_speed::VocalSpeedMod,
        voice_id::VoiceId,
    };

    /* STT */
    pub use crate::{
        mic_backend::MicBackend,
        noise_reduction::NoiseReduction,
        speech_source::SpeechSource,
        stt_conversation::{SttConversationBuilder, SttConversationExt},
        stt_engine::SttEngine,
        timestamps::{Diarization, Punctuation, TimestampsGranularity, WordTimestamps},
        transcript::{TranscriptSegment, TranscriptStream},
        vad_mode::VadMode,
    };
}
