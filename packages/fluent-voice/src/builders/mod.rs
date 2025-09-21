//! Builder implementations for the fluent voice API.
//!
//! This module provides concrete implementations of the various builder traits
//! in the fluent voice API. These can be used to create custom engines without
//! relying on macros.

// Re-export TTS builders
mod tts_builder;
pub use tts_builder::{
    builder::tts_conversation_builder, SpeakerLine, SpeakerLineBuilder, TtsConversationBuilderImpl,
    TtsConversationImpl,
};

// STT builders removed - orphaned code that was not used by the working DefaultSTTConversationBuilder system

// Re-export ElevenLabs extension builders
mod audio_isolation_builder;
mod sound_effects_builder;
mod speech_to_speech_builder;
mod voice_clone_builder;
mod voice_discovery_builder;

pub use audio_isolation_builder::{
    AudioIsolationBuilder, AudioIsolationBuilderImpl, AudioIsolationSession,
    AudioIsolationSessionImpl,
};
pub use sound_effects_builder::{
    SoundEffectsBuilder, SoundEffectsBuilderImpl, SoundEffectsSession, SoundEffectsSessionImpl,
};
pub use speech_to_speech_builder::{
    SpeechToSpeechBuilder, SpeechToSpeechBuilderImpl, SpeechToSpeechSession,
    SpeechToSpeechSessionImpl,
};
pub use voice_clone_builder::{VoiceCloneBuilder, VoiceCloneBuilderImpl};
pub use voice_discovery_builder::{VoiceDiscoveryBuilder, VoiceDiscoveryBuilderImpl};
