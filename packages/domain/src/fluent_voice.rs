//! Unified entry point trait for TTS and STT operations.

use crate::{
    audio_isolation::AudioIsolationBuilder,
    sound_effects::SoundEffectsBuilder,
    speech_to_speech::SpeechToSpeechBuilder,
    tts_conversation::TtsConversationBuilder,
    voice_clone::VoiceCloneBuilder,
    voice_discovery::VoiceDiscoveryBuilder,
    wake_word::{
        WakeWordBuilder, WakeWordConfig, WakeWordDetector, WakeWordEvent, WakeWordResult,
        WakeWordStream,
    },
};

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

    // STT functionality moved to fluent-voice implementation

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

/// Production-quality, zero-allocation wake word detector implementation.
/// Domain-only implementation that returns empty streams - concrete engines provide real functionality.
#[derive(Debug, Clone)]
pub struct DefaultWakeWordDetector {
    config: WakeWordConfig,
}

impl Default for DefaultWakeWordDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultWakeWordDetector {
    /// Create a new wake word detector with default configuration.
    #[inline]
    pub fn new() -> Self {
        Self {
            config: WakeWordConfig::default(),
        }
    }

    /// Create a new wake word detector with custom configuration.
    #[inline]
    pub fn with_config(config: WakeWordConfig) -> Self {
        Self { config }
    }
}

impl WakeWordDetector for DefaultWakeWordDetector {
    type Event = WakeWordEvent;

    #[inline]
    fn add_wake_word_model<P: AsRef<std::path::Path>>(
        &mut self,
        _model_path: P,
        _wake_word: String,
    ) -> WakeWordResult<()> {
        // Domain-only implementation - concrete engines handle model loading
        Ok(())
    }

    #[inline]
    fn process_audio(&mut self, _audio_data: &[u8]) -> WakeWordResult<Vec<Self::Event>> {
        // Domain-only implementation - concrete engines provide real detection
        Ok(Vec::new())
    }

    #[inline]
    fn process_samples(&mut self, _samples: &[f32]) -> WakeWordResult<Vec<Self::Event>> {
        // Domain-only implementation - concrete engines provide real detection
        Ok(Vec::new())
    }

    #[inline]
    fn update_config(&mut self, config: WakeWordConfig) -> WakeWordResult<()> {
        self.config = config;
        Ok(())
    }

    #[inline]
    fn get_config(&self) -> &WakeWordConfig {
        &self.config
    }
}

impl WakeWordStream for DefaultWakeWordDetector {
    type Event = WakeWordEvent;

    #[inline]
    fn process_stream<S>(
        &mut self,
        _audio_stream: S,
    ) -> impl futures_core::Stream<Item = WakeWordResult<Self::Event>> + Send
    where
        S: futures_core::Stream<Item = Vec<u8>> + Send + Unpin,
    {
        // Domain-only implementation - returns empty stream
        futures::stream::empty()
    }

    #[inline]
    fn process_sample_stream<S>(
        &mut self,
        _sample_stream: S,
    ) -> impl futures_core::Stream<Item = WakeWordResult<Self::Event>> + Send
    where
        S: futures_core::Stream<Item = Vec<f32>> + Send + Unpin,
    {
        // Domain-only implementation - returns empty stream
        futures::stream::empty()
    }
}

/// Production-quality, zero-allocation wake word builder implementation.
/// Domain-only implementation - concrete engines provide real functionality.
#[derive(Debug, Clone)]
pub struct DefaultWakeWordBuilder {
    config: WakeWordConfig,
}

impl Default for DefaultWakeWordBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultWakeWordBuilder {
    /// Create a new wake word builder with default configuration.
    #[inline]
    pub fn new() -> Self {
        Self {
            config: WakeWordConfig::default(),
        }
    }
}

impl WakeWordBuilder for DefaultWakeWordBuilder {
    type Detector = DefaultWakeWordDetector;

    #[inline]
    fn with_wake_word_model<P: AsRef<std::path::Path>>(
        self,
        _model_path: P,
        _wake_word: String,
    ) -> WakeWordResult<Self> {
        // Domain-only implementation - concrete engines handle model loading
        Ok(self)
    }

    #[inline]
    fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.config.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    #[inline]
    fn with_debug(mut self, debug: bool) -> Self {
        self.config.debug = debug;
        self
    }

    #[inline]
    fn build(self) -> WakeWordResult<Self::Detector> {
        Ok(DefaultWakeWordDetector::with_config(self.config))
    }
}

/// Default implementation entry point for FluentVoice
pub struct FluentVoiceImpl;

// Note: FluentVoice trait implementations should be provided by the fluent-voice crate,
// not in the domain crate. The domain crate only contains trait definitions.

// DummySegment removed - only real production transcript types allowed
// Real TtsChunk from Whisper crate is used throughout the codebase

// Extension trait implementations should be provided by the fluent-voice crate,
// not the domain crate. The domain crate only contains trait definitions.
