//! # Fluent Voice API
//!
//! Pure-trait fluent builder API for TTS & STT engines.
//!
//! This crate provides trait-based interfaces for Text-to-Speech (TTS) and
//! Speech-to-Text (STT) engines with a fluent builder pattern that returns
//! streams directly for real-time processing.
//!
//! ## Usage Pattern
//!
//! ### TTS (Text-to-Speech)
//!
//! ```ignore
//! let audio_stream = FluentVoice::tts()
//!     .with_speaker(
//!         Speaker::named("Bob")
//!             .with_speed_modifier(VocalSpeedMod(0.9))
//!             .speak("Hello, world!")
//!             .build()
//!     )
//!     .on_chunk(|chunk_result| match chunk_result {
//!         Ok(chunk) => chunk,
//!         Err(e) => BadAudioStreamSegment::new(e)
//!     })
//!     .synthesize();
//! ```
//!
//! ### STT (Speech-to-Text)
//!
//! ```ignore
//! // Live microphone transcription
//! let text_stream = MyEngine::stt()
//!     .with_microphone("default")
//!     .vad_mode(VadMode::Accurate)
//!     .on_chunk(|segment_result| match segment_result {
//!         Ok(segment) => segment.text().to_string(),
//!         Err(e) => String::new()
//!     })
//!     .listen();
//!
//! // File transcription
//! let text_stream = MyEngine::stt()
//!     .transcribe("audio.wav")
//!     .on_chunk(|segment_result| match segment_result {
//!         Ok(segment) => segment.text().to_string(),
//!         Err(e) => String::new()
//!     })
//!     .emit();
//! ```

#![cfg_attr(feature = "simd", feature(portable_simd))]

// Removed custom acceleration constraint - following candle's CPU-first philosophy
// CPU-only builds are fully supported and recommended for cross-platform compatibility

// Note: Default features include "microphone" so this should not trigger
#[cfg(all(
    not(feature = "microphone"),
    not(feature = "encodec"),
    not(feature = "mimi"),
    not(feature = "snac")
))]
compile_error!("At least one audio feature must be enabled: microphone, encodec, mimi, or snac");

/* ───── shared fundamentals ───── */
pub mod arrow_syntax;
pub mod async_stream_helpers;
pub mod audio_chunk;
pub mod audio_device_manager;
pub mod json_syntax_transform;
pub mod stream_ext;

/* ───── arrow syntax support for examples ───── */
pub mod arrow_syntax_impl;
pub mod example_macros;
pub mod syntax_transform;

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
pub mod audio_stream;
pub mod audio_stream_ext;
pub mod builders;

/* ───── STT conversation trait definitions ───── */
pub mod stt_conversation;

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

    /* STT and TTS builder traits */
    pub use crate::stt_conversation::{
        MicrophoneBuilder, SttConversationBuilder, SttPostChunkBuilder, TranscriptionBuilder,
    };
    pub use crate::tts_conversation::TtsConversationChunkBuilder;

    // For convenience, allow calling FluentVoice::stt() and FluentVoice::tts() directly
    pub use crate::fluent_voice::FluentVoiceImpl as FluentVoiceStaticMethods;

    /* cyrup-sugars macros for Ok => Err => syntax and JSON object syntax */
    pub use cyrup_sugars::macros::*;
    pub use cyrup_sugars::prelude::*;

    /* Arrow syntax transformation macros */
    pub use crate::{fv_match, listen_transform, on_chunk_transform, synthesize_transform};
    // Real production transcript segment type from Whisper crate
    pub use fluent_voice_whisper::TtsChunk;

    /* TTS method macros that enable arrow syntax */
    pub use crate::{tts_on_chunk, tts_synthesize};

    /* Builder implementations */
    pub use crate::builders::{
        MicrophoneBuilderImpl, SpeakerLine as Speaker, SpeakerLineBuilder,
        SttConversationBuilderImpl, SttConversationImpl, TranscriptImpl, TranscriptionBuilderImpl,
        TtsConversationBuilderImpl, TtsConversationImpl, stt_conversation_builder,
        tts_conversation_builder,
    };

    /* Domain traits needed for examples */
    pub use fluent_voice_domain::{
        SpeakerBuilder, SpeakerExt, TtsConversation, TtsConversationBuilder,
    };

    /* Implement SpeakerExt for the Speaker alias to enable Speaker::speaker() syntax */
    impl fluent_voice_domain::SpeakerExt for crate::builders::SpeakerLine {
        fn speaker(name: impl Into<String>) -> impl fluent_voice_domain::SpeakerBuilder {
            crate::builders::SpeakerLineBuilder::speaker(name)
        }
    }

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
pub use fluent_voice_domain::{TranscriptionSegment, TranscriptionStream};
pub use fluent_voice_whisper::TtsChunk;
