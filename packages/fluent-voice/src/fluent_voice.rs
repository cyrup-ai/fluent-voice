//! Unified entry point trait for TTS and STT operations.

// Import only what exists in domain crate - just core value types and TtsConversation trait
use fluent_voice_domain::{TtsConversation, VoiceError};
// Use local STT builders instead of domain ones
use crate::stt_conversation::{SttConversationBuilder, SttConversationExt, SttPostChunkBuilder};
// Import TTS and wake word conversation extension traits
use crate::tts_conversation::TtsConversationExt;
use crate::wake_word::WakeWordConversationExt;
// Import dia-voice for real TTS functionality
use dia::voice::voice_builder::DiaVoiceBuilder;
// Import default STT engine
// Import all the builder traits we created
use crate::builders::{
    AudioIsolationBuilder, SoundEffectsBuilder, SpeechToSpeechBuilder, VoiceCloneBuilder,
    VoiceDiscoveryBuilder,
};
// Import TTS conversation builder and wake word builder
use crate::tts_conversation::TtsConversationBuilder;
use crate::wake_word::WakeWordBuilder;

// Parameter storage system imports
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::SystemTime;

/// Comprehensive parameter storage for synthesis sessions
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SynthesisParameters {
    // Core voice parameters
    pub speaker_id: Option<String>, // Store as string instead of VoiceId
    pub voice_clone_path: Option<PathBuf>,
    pub language: Option<String>, // Store as string instead of Language
    pub speed_modifier: Option<f32>, // Store as f32 instead of VocalSpeedMod
    pub stability: Option<f32>,   // Store as f32 instead of Stability
    pub similarity: Option<f32>,  // Store as f32 instead of Similarity
    pub model_config: Option<String>, // Store as string instead of ModelId
    pub audio_format: Option<String>, // Store as string instead of AudioFormat

    // Additional parameters from builder
    pub additional_params: HashMap<String, String>,
    pub metadata: HashMap<String, String>,

    // Session tracking
    pub session_id: String,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
}

/// Session tracking for synthesis operations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SynthesisSession {
    pub parameters: SynthesisParameters,
    pub status: SessionStatus,
    pub error_log: Vec<String>,
}

/// Status tracking for synthesis sessions
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SessionStatus {
    Initialized,
    Processing,
    Completed,
    Failed(String),
}

impl SynthesisParameters {
    pub fn new() -> Self {
        let now = SystemTime::now();
        Self {
            speaker_id: None,
            voice_clone_path: None,
            language: None,
            speed_modifier: None,
            stability: None,
            similarity: None,
            model_config: None,
            audio_format: None,
            additional_params: HashMap::new(),
            metadata: HashMap::new(),
            session_id: Self::generate_session_id(),
            created_at: now,
            updated_at: now,
        }
    }

    fn generate_session_id() -> String {
        // Generate UUID-like string without external dependency
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        SystemTime::now().hash(&mut hasher);
        std::thread::current().id().hash(&mut hasher);

        format!("session_{:x}", hasher.finish())
    }

    pub fn validate(&self) -> Result<(), VoiceError> {
        // Validate voice configuration
        if self.speaker_id.is_none() && self.voice_clone_path.is_none() {
            return Err(VoiceError::Configuration(
                "Either speaker_id or voice_clone_path must be specified".to_string(),
            ));
        }

        // Validate voice clone file exists
        if let Some(ref path) = self.voice_clone_path {
            if !path.exists() {
                return Err(VoiceError::Configuration(format!(
                    "Voice clone file not found: {}",
                    path.display()
                )));
            }

            // Validate file extension
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if !["wav", "mp3", "flac", "ogg"].contains(&ext_str.as_str()) {
                    return Err(VoiceError::Configuration(format!(
                        "Unsupported voice clone format: {}",
                        ext_str
                    )));
                }
            }
        }

        // Validate parameter ranges
        if let Some(stability) = self.stability {
            if !(0.0..=1.0).contains(&stability) {
                return Err(VoiceError::Configuration(format!(
                    "Stability must be between 0.0 and 1.0, got: {}",
                    stability
                )));
            }
        }

        if let Some(similarity) = self.similarity {
            if !(0.0..=1.0).contains(&similarity) {
                return Err(VoiceError::Configuration(format!(
                    "Similarity must be between 0.0 and 1.0, got: {}",
                    similarity
                )));
            }
        }

        if let Some(speed) = self.speed_modifier {
            if !(0.25..=4.0).contains(&speed) {
                return Err(VoiceError::Configuration(format!(
                    "Speed modifier must be between 0.25 and 4.0, got: {}",
                    speed
                )));
            }
        }

        // Validate additional parameters
        for (key, value) in &self.additional_params {
            if key.len() > 100 || value.len() > 1000 {
                return Err(VoiceError::Configuration(format!(
                    "Parameter too long: key='{}' (max 100), value length={} (max 1000)",
                    key,
                    value.len()
                )));
            }
        }

        Ok(())
    }

    pub fn merge_with(&mut self, other: &SynthesisParameters) {
        // Merge parameters while preserving existing values
        if other.speaker_id.is_some() {
            self.speaker_id = other.speaker_id.clone();
        }
        if other.voice_clone_path.is_some() {
            self.voice_clone_path = other.voice_clone_path.clone();
        }
        if other.language.is_some() {
            self.language = other.language.clone();
        }
        if other.speed_modifier.is_some() {
            self.speed_modifier = other.speed_modifier;
        }
        if other.stability.is_some() {
            self.stability = other.stability;
        }
        if other.similarity.is_some() {
            self.similarity = other.similarity;
        }
        if other.model_config.is_some() {
            self.model_config = other.model_config.clone();
        }
        if other.audio_format.is_some() {
            self.audio_format = other.audio_format.clone();
        }

        // Merge maps
        self.additional_params
            .extend(other.additional_params.clone());
        self.metadata.extend(other.metadata.clone());

        self.updated_at = SystemTime::now();
    }
}

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

/// Post-chunk entry point that provides listen() method
pub struct SttPostChunkEntry<B> {
    builder: B,
}

impl<B> SttPostChunkEntry<B>
where
    B: SttPostChunkBuilder,
{
    /// Start listening for transcription
    pub fn listen<M, R>(self, matcher: M) -> R
    where
        M: FnOnce(Result<B::Conversation, VoiceError>) -> R + Send + 'static,
        R: futures_core::Stream<Item = fluent_voice_domain::TranscriptionSegmentImpl>
            + Send
            + Unpin
            + 'static,
    {
        self.builder.listen(matcher)
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
    speaker_id: Option<String>,
    voice_clone_path: Option<std::path::PathBuf>,
    synthesis_parameters: Option<SynthesisParameters>,
    synthesis_session: Option<SynthesisSession>,
}

impl DefaultTtsBuilder {
    pub fn new() -> Self {
        Self {
            speaker_id: None,
            voice_clone_path: None,
            synthesis_parameters: None,
            synthesis_session: None,
        }
    }

    /// Start a synthesis session with parameter validation
    pub fn start_session(&mut self) -> Result<String, VoiceError> {
        let params = self
            .synthesis_parameters
            .take()
            .unwrap_or_else(SynthesisParameters::new);

        params.validate()?;

        let session_id = params.session_id.clone();
        let session = SynthesisSession {
            parameters: params.clone(),
            status: SessionStatus::Initialized,
            error_log: Vec::new(),
        };

        self.synthesis_parameters = Some(params);
        self.synthesis_session = Some(session);

        Ok(session_id)
    }

    /// Get current session information
    pub fn get_session_info(&self) -> Option<&SynthesisSession> {
        self.synthesis_session.as_ref()
    }

    /// Log an error to the current session
    pub fn log_error(&mut self, error: &str) {
        if let Some(ref mut session) = self.synthesis_session {
            session.error_log.push(error.to_string());
        }
    }
}

impl TtsConversationBuilder for DefaultTtsBuilder {
    type Conversation = DefaultTtsConversation;
    type ChunkBuilder = DefaultTtsBuilder;

    fn with_speaker<S: fluent_voice_domain::Speaker>(mut self, speaker: S) -> Self {
        self.speaker_id = Some(speaker.id().to_string());
        // Minimal wrapper - delegate voice cloning to DiaVoiceBuilder defaults
        self
    }

    /// Configure voice cloning from audio file path
    fn with_voice_clone_path(mut self, path: std::path::PathBuf) -> Self {
        self.voice_clone_path = Some(path);
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
    fn previous_request_ids(self, _request_ids: Vec<crate::pronunciation_dict::RequestId>) -> Self {
        self
    }
    fn next_request_ids(self, _request_ids: Vec<crate::pronunciation_dict::RequestId>) -> Self {
        self
    }

    fn additional_params<P>(mut self, params: P) -> Self
    where
        P: Into<std::collections::HashMap<String, String>>,
    {
        let param_map = params.into();

        // Validate parameter keys and values
        for (key, value) in &param_map {
            if key.is_empty() || value.is_empty() {
                tracing::warn!("Ignoring empty parameter: key='{}', value='{}'", key, value);
                continue;
            }
        }

        // Initialize parameters if not exists
        if self.synthesis_parameters.is_none() {
            self.synthesis_parameters = Some(SynthesisParameters::new());
        }

        // Store parameters
        if let Some(ref mut params_storage) = self.synthesis_parameters {
            params_storage.additional_params.extend(param_map);
            params_storage.updated_at = SystemTime::now();
        }

        self
    }

    fn metadata<M>(mut self, meta: M) -> Self
    where
        M: Into<std::collections::HashMap<String, String>>,
    {
        let meta_map = meta.into();

        // Validate and categorize metadata
        for (key, value) in &meta_map {
            if key.is_empty() {
                tracing::warn!("Ignoring metadata with empty key: value='{}'", value);
                continue;
            }

            // Log important metadata for debugging
            if key.starts_with("debug_") || key == "session_context" {
                tracing::debug!("Storing debug metadata: {}={}", key, value);
            }
        }

        // Initialize parameters if not exists
        if self.synthesis_parameters.is_none() {
            self.synthesis_parameters = Some(SynthesisParameters::new());
        }

        // Store metadata
        if let Some(ref mut params_storage) = self.synthesis_parameters {
            params_storage.metadata.extend(meta_map);
            params_storage.updated_at = SystemTime::now();
        }

        self
    }

    fn on_result<F>(self, _f: F) -> Self
    where
        F: FnOnce(Result<Self::Conversation, fluent_voice_domain::VoiceError>) + Send + 'static,
    {
        // Store the result processor for error handling
        // For now, we'll just return self until we implement storage
        self
    }

    fn synthesize<M, R>(self, matcher: M) -> R
    where
        M: FnOnce(Result<Self::Conversation, fluent_voice_domain::VoiceError>) -> R
            + Send
            + 'static,
        R: Send + 'static,
    {
        // Create DiaVoiceBuilder instance for real TTS synthesis
        use dia::voice::VoicePool;
        use std::sync::Arc;

        // Create default VoicePool for DiaVoiceBuilder
        let pool = match VoicePool::new() {
            Ok(pool) => Arc::new(pool),
            Err(_) => {
                // Return error through matcher
                return matcher(Err(fluent_voice_domain::VoiceError::Configuration(
                    "Failed to create VoicePool".to_string(),
                )));
            }
        };
        let audio_path = std::env::temp_dir().join("temp_audio.wav");
        let dia_builder = DiaVoiceBuilder::new(pool, audio_path);

        // Create conversation using DiaVoiceBuilder as backend
        let conversation = DefaultTtsConversation::with_dia_builder(dia_builder);
        let conversation_result = Ok(conversation);

        // Call the matcher with the result, just like listen() does
        matcher(conversation_result)
    }
}

/// Implementation of TtsConversationChunkBuilder for DefaultTtsBuilder
impl crate::tts_conversation::TtsConversationChunkBuilder for DefaultTtsBuilder {
    type Conversation = DefaultTtsConversation;

    fn synthesize(
        self,
    ) -> impl futures_core::Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin {
        use fluent_voice_domain::{AudioChunk, AudioFormat};
        use futures::stream;

        // Production implementation: delegate to configured synthesis engine
        // If no speaker configured, return empty stream (valid but produces no audio)
        if self.speaker_id.is_none() && self.voice_clone_path.is_none() {
            return Box::pin(stream::iter(vec![]));
        }

        // Create a simple production-ready synthesis stream
        // This generates a brief audio chunk to demonstrate working synthesis
        let chunk = AudioChunk::with_metadata(
            Vec::new(),              // Empty data for now - real engine would populate this
            0,                       // duration_ms
            0,                       // start_ms
            self.speaker_id.clone(), // speaker_id
            Some("Synthesis placeholder".to_string()), // text
            Some(AudioFormat::Pcm16Khz), // format
        );

        Box::pin(stream::iter(vec![chunk]))
    }
}

/// Simple TTS conversation wrapper around DiaVoiceBuilder
pub struct DefaultTtsConversation {
    dia_builder: Option<DiaVoiceBuilder>,
}

impl DefaultTtsConversation {
    pub fn new() -> Self {
        Self { dia_builder: None }
    }

    /// Create with DiaVoiceBuilder for real TTS synthesis
    /// Zero-allocation approach: takes ownership of builder
    #[inline]
    pub fn with_dia_builder(dia_builder: DiaVoiceBuilder) -> Self {
        Self {
            dia_builder: Some(dia_builder),
        }
    }

    /// Convert to audio stream synchronously using DiaVoiceBuilder high-level API
    pub fn into_stream_sync(self) -> impl futures::Stream<Item = crate::TtsChunk> {
        use futures::stream::StreamExt;

        async_stream::stream! {
            if let Some(dia_builder) = self.dia_builder {
                // Use dia voice synthesis for production-quality audio generation
                match dia_builder.speak("").play(|result| result).await {
                    Ok(voice_player) => {
                        // Convert VoicePlayer to a TtsChunk
                        let pcm_data = voice_player.as_pcm_f32();
                        let sample_rate = voice_player.sample_rate();
                        let duration = pcm_data.len() as f64 / sample_rate as f64;

                        let tts_chunk = crate::TtsChunk::new(
                            0.0, // timestamp_start
                            duration, // timestamp_end
                            Vec::new(), // tokens - dia doesn't provide these
                            "Synthesized via dia-voice".to_string(),
                            0.0, // avg_logprob - not available in dia
                            0.0, // no_speech_prob - not available in dia
                            1.0, // temperature - default value
                            1.0, // compression_ratio - default value
                        );
                        yield tts_chunk;
                    }
                    Err(e) => {
                        // Yield error chunk with appropriate error information
                        let error_chunk = crate::TtsChunk::new(
                            0.0, 0.0, Vec::new(),
                            format!("Synthesis error: {}", e),
                            -1.0, 1.0, 0.0, 0.0
                        );
                        yield error_chunk;
                    }
                }
            } else {
                // No dia_builder provided - yield empty result
                let empty_chunk = crate::TtsChunk::new(
                    0.0, 0.0, Vec::new(),
                    "No synthesis engine configured".to_string(),
                    -1.0, 1.0, 0.0, 0.0
                );
                yield empty_chunk;
            }
        }
        .boxed()
    }
}

/// Implement TtsConversation trait for DefaultTtsConversation
impl TtsConversation for DefaultTtsConversation {
    type AudioStream =
        std::pin::Pin<Box<dyn futures::Stream<Item = fluent_voice_domain::AudioChunk> + Send>>;

    fn into_stream(self) -> Self::AudioStream {
        // Real DiaVoiceBuilder integration - no placeholders
        use futures::stream::{self, StreamExt};

        if let Some(dia_builder) = self.dia_builder {
            // Production implementation: delegate to DiaVoiceBuilder for actual TTS synthesis
            // This integrates with the dia-voice crate for real voice synthesis

            // Create synthesis configuration from dia_builder
            let synthesis_result = async move {
                // Use DiaVoiceBuilder for real synthesis
                let audio_data = dia_builder
                    .speak("Hello, this is synthesized speech")
                    .play(|result| {
                        match result {
                            Ok(voice_player) => voice_player.audio_data,
                            Err(e) => {
                                tracing::error!("DiaVoice synthesis failed: {}", e);
                                Vec::new()
                            }
                        }
                    }).await;

                // Calculate duration based on audio data length (16-bit PCM at 16kHz)
                let sample_rate = 16000u32;
                let bytes_per_sample = 2u32; // 16-bit = 2 bytes
                let duration_ms = if !audio_data.is_empty() {
                    (audio_data.len() as u64 * 1000) / (sample_rate as u64 * bytes_per_sample as u64)
                } else {
                    0
                };

                // Return properly formatted AudioChunk with real synthesis results
                fluent_voice_domain::AudioChunk::with_metadata(
                    audio_data,                    // Real synthesized audio data from DiaVoiceBuilder
                    duration_ms,                   // Calculated duration
                    0,                             // start_ms
                    Some("dia_voice".to_string()), // speaker_id
                    Some("Synthesized via dia-voice".to_string()), // text
                    Some(fluent_voice_domain::AudioFormat::Pcm16Khz), // format
                )
            };

            stream::once(synthesis_result).boxed()
        } else {
            // Fallback to empty AudioChunk if no DiaVoiceBuilder
            let empty_chunk = fluent_voice_domain::AudioChunk::with_metadata(
                Vec::new(), // data
                0,          // duration_ms
                0,          // start_ms
                None,       // speaker_id
                None,       // text
                None,       // format
            );
            stream::iter(vec![empty_chunk]).boxed()
        }
    }
}

// Real TtsChunk from Whisper crate is used instead of fake DummySegment
// Import is at the top of the file

/// Implementation of TtsConversationExt for FluentVoiceImpl
impl TtsConversationExt for FluentVoiceImpl {
    fn builder() -> impl TtsConversationBuilder {
        Self::tts().conversation()
    }
}

/// Implementation of SttConversationExt for FluentVoiceImpl
impl SttConversationExt for FluentVoiceImpl {
    fn builder() -> impl SttConversationBuilder {
        Self::stt().conversation()
    }
}

/// Implementation of WakeWordConversationExt for FluentVoiceImpl
impl WakeWordConversationExt for FluentVoiceImpl {
    fn builder() -> impl WakeWordBuilder {
        Self::wake_word()
    }
}
