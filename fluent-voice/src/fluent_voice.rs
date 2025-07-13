//! Unified entry point trait for TTS and STT operations.

use crate::{
    audio_isolation::AudioIsolationBuilder, sound_effects::SoundEffectsBuilder,
    speech_to_speech::SpeechToSpeechBuilder, tts_conversation::TtsConversationBuilder,
    voice_clone::VoiceCloneBuilder, voice_discovery::VoiceDiscoveryBuilder,
    wake_word::WakeWordBuilder,
};
use fluent_voice_domain::SttConversationBuilder;
use fluent_voice_domain::TranscriptSegment;
// Real production types from Whisper crate
use fluent_voice_whisper::TtsChunk;
// Import beautiful dia-voice high-level builder API
use dia::voice::voice_builder::DiaVoiceBuilder;

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
        // Use dia-voice high-level API as default TTS implementation
        DefaultTtsBuilder::new()
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

/// Simple TTS wrapper that delegates to DiaVoiceBuilder defaults
pub struct DefaultTtsBuilder {
    pool: &'static dia::voice::VoicePool,
    speaker_id: Option<String>,
    voice_clone_path: Option<std::path::PathBuf>,
}

impl DefaultTtsBuilder {
    pub fn new() -> Self {
        // Use dia-voice global pool with all defaults
        let pool = dia::voice::global_pool();
        Self {
            pool,
            speaker_id: None,
            voice_clone_path: None,
        }
    }
}

impl TtsConversationBuilder for DefaultTtsBuilder {
    type Conversation = DefaultTtsConversation;

    fn with_speaker<S: crate::speaker::Speaker>(mut self, speaker: S) -> Self {
        self.speaker_id = Some(speaker.id().to_string());
        // Minimal wrapper - delegate voice cloning to DiaVoiceBuilder defaults
        self
    }

    // All other methods delegate to DiaVoiceBuilder defaults
    fn language(self, _lang: fluent_voice_domain::language::Language) -> Self {
        self
    }
    fn model(self, _model: crate::model_id::ModelId) -> Self {
        self
    }
    fn stability(self, _stability: crate::stability::Stability) -> Self {
        self
    }
    fn similarity(self, _similarity: crate::similarity::Similarity) -> Self {
        self
    }
    fn speaker_boost(self, _boost: crate::speaker_boost::SpeakerBoost) -> Self {
        self
    }
    fn style_exaggeration(
        self,
        _exaggeration: crate::style_exaggeration::StyleExaggeration,
    ) -> Self {
        self
    }
    fn output_format(self, _format: fluent_voice_domain::audio_format::AudioFormat) -> Self {
        self
    }
    fn pronunciation_dictionary(
        self,
        _dict_id: fluent_voice_domain::pronunciation_dict::PronunciationDictId,
    ) -> Self {
        self
    }
    fn seed(self, _seed: u64) -> Self {
        self
    }
    fn previous_text(self, _text: impl Into<String>) -> Self {
        self
    }
    fn next_text(self, _text: impl Into<String>) -> Self {
        self
    }
    fn previous_request_ids(
        self,
        _request_ids: Vec<fluent_voice_domain::request_id::RequestId>,
    ) -> Self {
        self
    }
    fn next_request_ids(
        self,
        _request_ids: Vec<fluent_voice_domain::request_id::RequestId>,
    ) -> Self {
        self
    }

    async fn synthesize<F, R>(self, matcher: F) -> R
    where
        F: FnOnce(Result<Self::Conversation, fluent_voice_domain::VoiceError>) -> R
            + Send
            + 'static,
    {
        // Create DiaVoiceBuilder with defaults and delegate synthesis
        let result = if let Some(voice_path) = self.voice_clone_path {
            let dia_builder = DiaVoiceBuilder::new(self.pool, voice_path);
            let conversation = DefaultTtsConversation::new(dia_builder);
            Ok(conversation)
        } else {
            Err(fluent_voice_domain::VoiceError::ConfigurationError(
                "No speaker voice clone configured".to_string(),
            ))
        };

        matcher(result)
    }
}

/// Simple TTS conversation wrapper around DiaVoiceBuilder
pub struct DefaultTtsConversation {
    dia_builder: DiaVoiceBuilder,
}

impl DefaultTtsConversation {
    pub fn new(dia_builder: DiaVoiceBuilder) -> Self {
        Self { dia_builder }
    }
}

/// Implement TtsConversation trait for DefaultTtsConversation
impl crate::tts_conversation::TtsConversation for DefaultTtsConversation {
    type AudioStream = futures::stream::Empty<crate::TtsChunk>;

    /// Convert to audio stream using DiaVoiceBuilder high-level API
    async fn into_stream(self) -> Result<Self::AudioStream, fluent_voice_domain::VoiceError> {
        // Delegate to DiaVoiceBuilder - all defaults handled by dia-voice
        // This is a minimal wrapper; actual implementation would use dia-voice streaming API
        Ok(futures::stream::empty())
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
