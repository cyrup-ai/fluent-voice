//! Unified entry point trait for TTS and STT operations.

use crate::{
    audio_isolation::AudioIsolationBuilder, sound_effects::SoundEffectsBuilder,
    speech_to_speech::SpeechToSpeechBuilder,
    tts_conversation::TtsConversationBuilder,
    voice_clone::VoiceCloneBuilder, voice_discovery::VoiceDiscoveryBuilder,
    wake_word::WakeWordBuilder,
};
use fluent_voice_domain::SttConversationBuilder;
use fluent_voice_domain::TranscriptSegment;
// Real production types from Whisper crate
use fluent_voice_whisper::TtsChunk;
// Import dia-voice crate for TTS functionality
use dia::voice::dia_speaker::DiaSpeakerBuilder;

/// Unified entry point for Text-to-Speech and Speech-to-Text operations.
///
/// This trait provides the main entry points for the fluent voice API, allowing
/// users to start TTS or STT operations with a consistent interface.
///
/// # Examples
///
/// ```ignore
/// use fluent_voice::prelude::*;
///
/// // TTS usage
/// let audio = MyEngine::tts()
///     .with_speaker(
///         Speaker::named("Alice")
///             .speak("Hello, world!")
///             .build()
///     )
///     .synthesize(|conversation| {
///         Ok => conversation.into_stream(),
///         Err(e) => Err(e),
///     })
///     .await?;
///
/// // STT microphone usage
/// let stream = MyEngine::stt()
///     .with_microphone("default")
///     .listen(|conversation| {
///         Ok => conversation.into_stream(),
///         Err(e) => Err(e),
///     })
///     .await?;
///
/// // STT file transcription usage
/// let transcript = MyEngine::stt()
///     .transcribe("audio.wav")
///     .emit(|transcript| {
///         Ok => transcript.into_stream(),
///         Err(e) => Err(e),
///     })
///     .await?;
/// ```
pub trait FluentVoice {
    /// Begin a new TTS conversation builder.
    ///
    /// This method returns a conversation builder that can be used to configure
    /// speakers, voice settings, and other TTS parameters before synthesis.
    ///
    /// # Returns
    ///
    /// A new TTS conversation builder instance.
    fn tts() -> impl TtsConversationBuilder;

    /// Begin a new STT conversation builder.
    ///
    /// This method returns a conversation builder that can be used to configure
    /// audio sources, language hints, VAD settings, and other recognition
    /// parameters before starting transcription.
    ///
    /// # Returns
    ///
    /// A new STT conversation builder instance.
    fn stt() -> impl SttConversationBuilder;

    /// Begin a new wake word detection builder.
    ///
    /// This method returns a builder that can be used to configure
    /// wake word models, confidence thresholds, and other detection
    /// parameters before starting wake word detection.
    ///
    /// # Returns
    ///
    /// A new wake word builder instance.
    fn wake_word() -> impl WakeWordBuilder;

    /// Begin a new voice discovery builder.
    ///
    /// This method returns a builder that can be used to search and
    /// filter available voices from the engine provider.
    ///
    /// # Returns
    ///
    /// A new voice discovery builder instance.
    fn voices() -> impl VoiceDiscoveryBuilder;

    /// Begin a new voice cloning builder.
    ///
    /// This method returns a builder that can be used to create
    /// custom voices from audio samples.
    ///
    /// # Returns
    ///
    /// A new voice cloning builder instance.
    fn clone_voice() -> impl VoiceCloneBuilder;

    /// Begin a new speech-to-speech conversion builder.
    ///
    /// This method returns a builder that can be used to convert
    /// speech from one voice to another while preserving characteristics.
    ///
    /// # Returns
    ///
    /// A new speech-to-speech builder instance.
    fn speech_to_speech() -> impl SpeechToSpeechBuilder;

    /// Begin a new audio isolation builder.
    ///
    /// This method returns a builder that can be used to separate
    /// voices from background audio or isolate specific audio components.
    ///
    /// # Returns
    ///
    /// A new audio isolation builder instance.
    fn audio_isolation() -> impl AudioIsolationBuilder;

    /// Begin a new sound effects generation builder.
    ///
    /// This method returns a builder that can be used to generate
    /// audio effects from text descriptions.
    ///
    /// # Returns
    ///
    /// A new sound effects builder instance.
    fn sound_effects() -> impl SoundEffectsBuilder;
}

/// Default implementation entry point for FluentVoice
pub struct FluentVoiceImpl;

impl FluentVoice for FluentVoiceImpl {
    fn tts() -> impl TtsConversationBuilder {
        // Use dia-voice as the canonical default TTS provider
        DiaSpeakerBuilder::new("default_speaker".to_string())
    }

    fn stt() -> impl SttConversationBuilder {
        // Use DefaultSTTEngine with canonical providers (Whisper, VAD, Koffee)
        crate::engines::DefaultSTTConversationBuilder::new()
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

// Real TtsChunk from Whisper crate is used instead of fake DummySegment
// Import is at the top of the file

/// Implementation of TtsConversationExt for FluentVoiceImpl
impl crate::tts_conversation::TtsConversationExt for FluentVoiceImpl {
    fn builder() -> impl TtsConversationBuilder {
        Self::tts()
    }
}

/// Implementation of SttConversationExt for FluentVoiceImpl
impl crate::stt_conversation::SttConversationExt for FluentVoiceImpl {
    fn builder() -> impl SttConversationBuilder {
        Self::stt()
    }
}

/// Implementation of WakeWordConversationExt for FluentVoiceImpl
impl crate::wake_word_conversation::WakeWordConversationExt for FluentVoiceImpl {
    fn builder() -> impl WakeWordBuilder {
        Self::wake_word()
    }
}
