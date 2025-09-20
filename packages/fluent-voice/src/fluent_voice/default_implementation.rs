//! Default implementation of FluentVoice trait.

use super::entry_points::{SttEntry, TtsEntry};
use super::fluent_voice_trait::FluentVoice;
use crate::builders::{
    AudioIsolationBuilder, SoundEffectsBuilder, SpeechToSpeechBuilder, VoiceCloneBuilder,
    VoiceDiscoveryBuilder,
};
use crate::wake_word::WakeWordBuilder;

/// Default implementation entry point for FluentVoice
pub struct FluentVoiceImpl;

impl FluentVoice for FluentVoiceImpl {
    fn tts() -> TtsEntry {
        TtsEntry::new()
    }

    fn stt() -> SttEntry {
        SttEntry::new()
    }

    fn wake_word() -> impl WakeWordBuilder {
        // Use Koffee as the default wake word implementation
        crate::wake_word_koffee::KoffeeWakeWordBuilder::new()
    }

    fn voices() -> impl VoiceDiscoveryBuilder {
        crate::builders::VoiceDiscoveryBuilderImpl::new()
    }

    fn clone_voice() -> impl VoiceCloneBuilder {
        crate::builders::VoiceCloneBuilderImpl::new()
    }

    fn speech_to_speech() -> impl SpeechToSpeechBuilder {
        crate::builders::SpeechToSpeechBuilderImpl::new()
    }

    fn audio_isolation() -> impl AudioIsolationBuilder {
        crate::builders::AudioIsolationBuilderImpl::new()
    }

    fn sound_effects() -> impl SoundEffectsBuilder {
        crate::builders::SoundEffectsBuilderImpl::new()
    }
}
