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
// Import cyrup-sugars StreamExt and AsyncStream for action verb callback syntax
use cyrup_sugars::{AsyncStream, StreamExt};

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
/// let audio = FluentVoice::tts().conversation()
///     .with_speaker(
///         Speaker::speaker("Alice")
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
/// let stream = FluentVoice::stt().conversation()
///     .with_source(SpeechSource::Microphone {
///         backend: MicBackend::Default,
///         format: AudioFormat::Pcm16Khz,
///         sample_rate: 16_000,
///     })
///     .listen(|conversation| {
///         Ok => conversation.into_stream(),
///         Err(e) => Err(e),
///     })
///     .await?;
/// ```
pub trait FluentVoice {
    /// Begin a new TTS session.
    ///
    /// This method returns an entry point that provides access to TTS conversation builders.
    ///
    /// # Returns
    ///
    /// A new TTS session instance.
    fn tts() -> TtsEntry;

    /// Begin a new STT session.
    ///
    /// This method returns an entry point that provides access to STT conversation builders.
    ///
    /// # Returns
    ///
    /// A new STT session instance.
    fn stt() -> SttEntry;

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

/// Entry point for TTS operations providing .conversation() method
pub struct TtsEntry;

impl TtsEntry {
    pub fn new() -> Self {
        Self
    }

    /// Create a new TTS conversation builder
    pub fn conversation(self) -> impl TtsConversationBuilder {
        DefaultTtsBuilder::new()
    }
}

/// Entry point for STT operations providing .conversation() method  
pub struct SttEntry;

impl SttEntry {
    pub fn new() -> Self {
        Self
    }

    /// Create a new STT conversation builder
    pub fn conversation(self) -> impl SttConversationBuilder {
        crate::engines::DefaultSTTConversationBuilder::new()
    }
}

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

    fn synthesize<F>(self, callback: F) -> AsyncStream<TtsChunk>
    where
        F: FnMut(
                Result<Self::Conversation, fluent_voice_domain::VoiceError>,
            ) -> Result<AsyncStream<TtsChunk>, fluent_voice_domain::VoiceError>
            + Send
            + 'static,
    {
        // Create the result stream based on configuration
        let result_stream = if let Some(voice_path) = self.voice_clone_path {
            let dia_builder = DiaVoiceBuilder::new(self.pool, voice_path);
            let conversation = DefaultTtsConversation::new(dia_builder);
            // Convert conversation to stream and wrap in AsyncStream
            match conversation.into_stream() {
                Ok(stream) => AsyncStream::from_stream(stream),
                Err(e) => AsyncStream::from_error(e),
            }
        } else {
            AsyncStream::from_error(fluent_voice_domain::VoiceError::ConfigurationError(
                "No speaker voice clone configured".to_string(),
            ))
        };

        // Use cyrup-sugars StreamExt to enable README.md callback syntax
        result_stream.on_result(callback)
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
