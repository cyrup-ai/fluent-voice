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

/* ───── shared fundamentals ───── */
pub mod audio_device_manager;
pub mod language;

/* ───── TTS chain ───── */
pub mod model_id;
pub mod pitch_range;
pub mod similarity;
pub mod speaker;
pub mod speaker_boost;
pub mod speaker_builder;
pub mod stability;
pub mod style_exaggeration;
pub mod tts_conversation;
pub mod tts_engine;
pub mod tts_settings;
pub mod vocal_speed;
pub mod voice_id;

/* ───── STT chain ───── */
pub mod mic_backend;
pub mod noise_reduction;
pub mod stt_conversation;
pub mod stt_engine;
pub mod timestamps;
pub mod transcript;
pub mod vad_mode;

/* ───── ElevenLabs extensions ───── */
pub mod audio_isolation;
pub mod pronunciation_dict;
pub mod sound_effects;
pub mod speech_to_speech;
pub mod voice_clone;
pub mod voice_discovery;
pub mod voice_labels;

/* ───── wake word detection ───── */
pub mod wake_word;
pub mod wake_word_conversation;
pub mod wake_word_engine;
pub mod wake_word_koffee;

/* ───── audio input ───── */
pub mod audio_io;
pub use audio_io::AudioInput;

/* ───── internal matcher macro ───── */
mod macros;

/* ───── concrete builder implementations ───── */
pub mod builders;

/* ───── unified entry point ───── */
pub mod fluent_voice;

/* ───── production engine implementations ───── */
pub mod engines;

/* ───── prelude for users ───── */
pub mod prelude {
    //! Re-exports of commonly used types and traits.

    /* shared */
    pub use crate::language::Language;
    pub use fluent_voice_domain::{AudioFormat, SpeechSource, VoiceError};

    /* TTS */
    pub use crate::{
        model_id::ModelId,
        pitch_range::PitchRange,
        speaker_builder::SpeakerBuilder,
        tts_conversation::{TtsConversation, TtsConversationBuilder, TtsConversationExt},
        tts_engine::TtsEngine,
        tts_settings::{Similarity, SpeakerBoost, Stability, StyleExaggeration},
        vocal_speed::VocalSpeedMod,
        voice_id::VoiceId,
    };

    /* STT */
    pub use crate::{
        mic_backend::MicBackend,
        noise_reduction::NoiseReduction,
        // speech_source::SpeechSource, // Now imported from fluent_voice_domain
        stt_conversation::{
            MicrophoneBuilder, SttConversation, SttConversationBuilder, SttConversationExt,
            TranscriptionBuilder,
        },
        stt_engine::SttEngine,
        timestamps::{Diarization, Punctuation, TimestampsGranularity, WordTimestamps},
        transcript::{TranscriptSegment, TranscriptStream},
        vad_mode::VadMode,
    };

    /* Unified entry point */
    pub use crate::fluent_voice::{
        FluentVoice as FluentVoiceTrait, FluentVoiceImpl as FluentVoice,
    };
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
        wake_word::{
            WakeWordBuilder, WakeWordConfig, WakeWordDetector, WakeWordEvent, WakeWordStream,
        },
        wake_word_conversation::WakeWordConversationExt,
        wake_word_engine::WakeWordEngine,
        wake_word_koffee::{KoffeeWakeWordBuilder, KoffeeWakeWordDetector},
    };

    /* Engine trait implementations */

    /* ElevenLabs extensions */
    pub use crate::{
        audio_isolation::{AudioIsolationBuilder, AudioIsolationExt, AudioIsolationSession},
        pronunciation_dict::{PronunciationDictId, RequestId},
        sound_effects::{SoundEffectsBuilder, SoundEffectsExt, SoundEffectsSession},
        speech_to_speech::{SpeechToSpeechBuilder, SpeechToSpeechExt, SpeechToSpeechSession},
        voice_clone::{VoiceCloneBuilder, VoiceCloneExt, VoiceCloneResult},
        voice_discovery::{VoiceDiscoveryBuilder, VoiceDiscoveryExt, VoiceDiscoveryResult},
        voice_labels::{VoiceCategory, VoiceDetails, VoiceLabels, VoiceSettings, VoiceType},
    };
}
