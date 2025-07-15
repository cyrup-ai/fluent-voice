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
//! let audio = FluentVoice::tts()
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
//! // Live microphone transcription
//! let mut segments = MyEngine::stt()
//!     .with_microphone("default")
//!     .vad_mode(VadMode::Accurate)
//!     .listen(|conversation| {
//!         Ok => conversation.into_stream(),
//!         Err(e) => Err(e),
//!     })
//!     .await?;
//!
//! // File transcription
//! let transcript = MyEngine::stt()
//!     .transcribe("audio.wav")
//!     .emit(|transcript| {
//!         Ok => transcript.into_stream(),
//!         Err(e) => Err(e),
//!     })
//!     .await?;
//! ```

#![cfg_attr(feature = "simd", feature(portable_simd))]

#[cfg(not(any(
    feature = "cuda",
    feature = "metal",
    feature = "accelerate",
    feature = "mkl"
)))]
compile_error!(
    "At least one candle acceleration feature must be enabled: cuda, metal, accelerate, or mkl"
);

// Note: Default features include "microphone" so this should not trigger
#[cfg(all(
    not(feature = "microphone"),
    not(feature = "encodec"),
    not(feature = "mimi"),
    not(feature = "snac")
))]
compile_error!("At least one audio feature must be enabled: microphone, encodec, mimi, or snac");

/* ───── shared fundamentals ───── */
pub mod async_stream_helpers;
pub mod audio_chunk;
pub mod audio_device_manager;
pub mod stream_ext;

/* ───── wake word engine implementations ───── */
pub mod wake_word_engine;
pub mod wake_word_koffee;

/* ───── TTS settings aggregation ───── */
pub mod tts_settings;

/* ───── audio input ───── */
pub mod audio_io;
pub use audio_io::AudioInput;

/* ───── internal matcher macro ───── */
mod macros;

/* ───── concrete builder implementations ───── */
pub mod builders;

/* ───── unified entry point ───── */
pub mod fluent_voice;

/* ───── re-export domain types ───── */
pub use fluent_voice_domain::*;

/* ───── production engine implementations ───── */
pub mod engines;

/* ───── prelude for users ───── */
pub mod prelude {
    //! Re-exports of commonly used types and traits.

    /* Re-export everything from domain */
    pub use fluent_voice_domain::prelude::*;

    /* Fluent-voice specific implementations */
    pub use crate::{AsyncStream, AsyncTask};
    pub use crate::{
        audio_chunk::{AudioChunk, SynthesisChunk},
        stream_ext::TtsStreamExt,
        tts_settings::{Similarity, SpeakerBoost, Stability, StyleExaggeration},
    };

    /* Unified entry point */
    pub use crate::fluent_voice::{
        FluentVoice as FluentVoiceTrait, FluentVoiceImpl as FluentVoice, SttEntry, TtsEntry,
    };

    // For convenience, allow calling FluentVoice::stt() and FluentVoice::tts() directly
    pub use crate::fluent_voice::FluentVoiceImpl as FluentVoiceStaticMethods;

    /* cyrup-sugars macros for Ok => Err => syntax and JSON object syntax */
    pub use cyrup_sugars::prelude::*;
    pub use cyrup_sugars::macros::*;
    // Real production transcript segment type from Whisper crate
    pub use fluent_voice_whisper::TtsChunk;

    /* Builder implementations */
    pub use crate::builders::{
        MicrophoneBuilderImpl, SpeakerLine as Speaker, SpeakerLineBuilder,
        SttConversationBuilderImpl, SttConversationImpl, TranscriptImpl, TranscriptionBuilderImpl,
        TtsConversationBuilderImpl, TtsConversationImpl, stt_conversation_builder,
        tts_conversation_builder,
    };

    /* Wake Word Detection */
    pub use crate::{
        wake_word_engine::WakeWordEngine,
        wake_word_koffee::{KoffeeWakeWordBuilder, KoffeeWakeWordDetector},
    };

    /* Engine trait implementations */

    /* ElevenLabs extensions - fluent-voice concrete builders */
    pub use crate::builders::{
        AudioIsolationBuilderImpl, SoundEffectsBuilderImpl, SpeechToSpeechBuilderImpl,
        VoiceCloneBuilderImpl, VoiceDiscoveryBuilderImpl,
    };
}

// Re-export at crate root for internal use
pub use cyrup_sugars::{AsyncStream, AsyncTask};
pub use fluent_voice_whisper::TtsChunk;
pub use fluent_voice_domain::{TranscriptSegment, TranscriptStream};
