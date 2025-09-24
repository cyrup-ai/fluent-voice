//!
//! This module provides the FluentVoice trait implementation for ElevenLabs TTS engine.
//! It bridges the existing HTTP/3 QUIC engine with the fluent-voice trait system.

use crate::engine::{TtsEngine, TtsEngineBuilder, VoiceSettings as ElevenLabsVoiceSettings};
use async_stream;
use cyrup_sugars::prelude::MessageChunk;
use fluent_voice::fluent_voice::FluentVoice as FluentVoiceTrait;
use fluent_voice::fluent_voice::entry_points::SttEntry;
use fluent_voice::tts_conversation::{TtsConversationBuilder, TtsConversationChunkBuilder};
use fluent_voice::wake_word::WakeWordBuilder;
use fluent_voice_domain::TtsConversation;
use fluent_voice_domain::{
    AudioChunk, AudioFormat, Language, ModelId, PronunciationDictId, RequestId, Similarity,
    Speaker, SpeakerBoost, Stability, StyleExaggeration, VoiceError,
};
use futures_core::Stream;
use std::collections::HashMap;
use std::pin::Pin;

/// Result type for ElevenLabs operations
pub type Result<T> = std::result::Result<T, VoiceError>;

/// ElevenLabs implementation that shadows the FluentVoice trait
///
/// This struct provides the same API as FluentVoice but uses ElevenLabs internally
/// When users call `FluentVoice::tts()`, they get ElevenLabs functionality
pub struct FluentVoice;

/// Alternative ElevenLabs API for direct access
/// This provides async methods for advanced use cases
pub struct ElevenLabsFluentVoice;

impl FluentVoice {
    /// Static method for TTS that returns ElevenLabs conversation builder
    pub fn tts() -> ElevenLabsTtsEntryWrapper {
        ElevenLabsTtsEntryWrapper::new().unwrap_or_else(|e| {
            panic!(
                "Failed to initialize ElevenLabs TTS: {}. Ensure ELEVENLABS_API_KEY is set.",
                e
            )
        })
    }

    /// Static method for STT - delegates to default implementation
    pub fn stt() -> SttEntry {
        <fluent_voice::fluent_voice::FluentVoiceImpl as FluentVoiceTrait>::stt()
    }

    /// Wake word builder
    pub fn wake_word() -> impl WakeWordBuilder {
        fluent_voice::wake_word_koffee::KoffeeWakeWordBuilder::new()
    }
}

impl ElevenLabsFluentVoice {
    /// Async TTS method for the example
    pub async fn tts() -> Result<ElevenLabsTtsEntryWrapper> {
        ElevenLabsTtsEntryWrapper::new()
    }

    /// Static method for STT - delegates to default implementation
    pub fn stt() -> SttEntry {
        <fluent_voice::fluent_voice::FluentVoiceImpl as FluentVoiceTrait>::stt()
    }
}

/// Custom TtsEntry that uses ElevenLabs engine instead of default
pub struct ElevenLabsTtsEntryWrapper {
    engine: TtsEngine,
}

impl ElevenLabsTtsEntryWrapper {
    pub fn new() -> Result<Self> {
        // Create ElevenLabs engine with proper error handling
        let engine = TtsEngineBuilder::default()
            .api_key_from_env()
            .map_err(|e| VoiceError::Configuration(format!("Failed to get API key: {}", e)))?
            .http3_enabled(true)
            .build()
            .map_err(|e| VoiceError::Configuration(format!("Failed to build engine: {}", e)))?;

        Ok(Self { engine })
    }

    /// Create conversation builder that uses ElevenLabs
    pub fn conversation(self) -> ElevenLabsTtsConversationBuilder {
        ElevenLabsTtsConversationBuilder::new(self.engine)
    }
}

/// TTS conversation builder for ElevenLabs that uses the actual ElevenLabs engine
pub struct ElevenLabsTtsConversationBuilder {
    engine: TtsEngine,
    speakers: Vec<SpeakerData>,
    language: Option<Language>,
    model: Option<ModelId>,
    chunk_processor: Option<
        Box<dyn FnMut(std::result::Result<AudioChunk, VoiceError>) -> AudioChunk + Send + 'static>,
    >,
    stability: Option<Stability>,
    similarity: Option<Similarity>,
    speaker_boost: Option<SpeakerBoost>,
    style_exaggeration: Option<StyleExaggeration>,
    output_format: Option<AudioFormat>,
    pronunciation_dictionary: Option<PronunciationDictId>,
    seed: Option<u64>,
    previous_text: Option<String>,
    next_text: Option<String>,
    previous_request_ids: Vec<RequestId>,
    next_request_ids: Vec<RequestId>,
    additional_params: HashMap<String, String>,
    metadata: HashMap<String, String>,
}

/// Internal speaker data structure
#[derive(Debug, Clone)]
struct SpeakerData {
    id: String,
    text: String,
    voice_id: Option<String>,
    language: Option<Language>,
    speed_modifier: Option<f32>,
}

impl ElevenLabsTtsConversationBuilder {
    pub fn new(engine: TtsEngine) -> Self {
        Self {
            engine,
            speakers: Vec::new(),
            language: None,
            model: None,
            chunk_processor: None,
            stability: None,
            similarity: None,
            speaker_boost: None,
            style_exaggeration: None,
            output_format: None,
            pronunciation_dictionary: None,
            seed: None,
            previous_text: None,
            next_text: None,
            previous_request_ids: Vec::new(),
            next_request_ids: Vec::new(),
            additional_params: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}

impl TtsConversationBuilder for ElevenLabsTtsConversationBuilder {
    type Conversation = ElevenLabsTtsConversation;
    type ChunkBuilder = Self;

    fn with_speaker<S: Speaker>(mut self, speaker: S) -> Self {
        let speaker_data = SpeakerData {
            id: speaker.id().to_string(),
            text: speaker.text().to_string(),
            voice_id: speaker.voice_id().map(|v| v.to_string()),
            language: speaker.language().cloned(),
            speed_modifier: speaker.speed_modifier().map(|v| v.0),
        };
        self.speakers.push(speaker_data);
        self
    }

    fn with_voice_clone_path(self, _path: std::path::PathBuf) -> Self {
        self
    }

    fn language(mut self, lang: Language) -> Self {
        self.language = Some(lang);
        self
    }

    fn model(mut self, model: ModelId) -> Self {
        self.model = Some(model);
        self
    }

    fn stability(mut self, stability: Stability) -> Self {
        self.stability = Some(stability);
        self
    }

    fn similarity(mut self, similarity: Similarity) -> Self {
        self.similarity = Some(similarity);
        self
    }

    fn speaker_boost(mut self, boost: SpeakerBoost) -> Self {
        self.speaker_boost = Some(boost);
        self
    }

    fn style_exaggeration(mut self, exaggeration: StyleExaggeration) -> Self {
        self.style_exaggeration = Some(exaggeration);
        self
    }

    fn output_format(mut self, format: AudioFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    fn pronunciation_dictionary(mut self, dict_id: PronunciationDictId) -> Self {
        self.pronunciation_dictionary = Some(dict_id);
        self
    }

    fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn previous_text(mut self, text: impl Into<String>) -> Self {
        self.previous_text = Some(text.into());
        self
    }

    fn next_text(mut self, text: impl Into<String>) -> Self {
        self.next_text = Some(text.into());
        self
    }

    fn previous_request_ids(mut self, request_ids: Vec<RequestId>) -> Self {
        self.previous_request_ids = request_ids;
        self
    }

    fn next_request_ids(mut self, request_ids: Vec<RequestId>) -> Self {
        self.next_request_ids = request_ids;
        self
    }

    fn additional_params<P>(mut self, params: P) -> Self
    where
        P: Into<HashMap<String, String>>,
    {
        self.additional_params = params.into();
        self
    }

    fn metadata<M>(mut self, meta: M) -> Self
    where
        M: Into<HashMap<String, String>>,
    {
        self.metadata = meta.into();
        self
    }

    fn on_result<F>(self, _processor: F) -> Self
    where
        F: FnOnce(std::result::Result<Self::Conversation, VoiceError>) + Send + 'static,
    {
        self
    }

    fn on_chunk<F>(mut self, processor: F) -> Self::ChunkBuilder
    where
        F: FnMut(std::result::Result<AudioChunk, VoiceError>) -> AudioChunk + Send + 'static,
    {
        self.chunk_processor = Some(Box::new(processor));
        self
    }

    fn synthesize<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(std::result::Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: Stream<Item = AudioChunk> + Send + Unpin + 'static,
    {
        let conversation = ElevenLabsTtsConversation {
            speakers: self.speakers,
            language: self.language,
            engine: self.engine,
            chunk_processor: self.chunk_processor,
            stability: self.stability,
            similarity: self.similarity,
            style_exaggeration: self.style_exaggeration,
            speaker_boost: self.speaker_boost,
        };

        matcher(Ok(conversation))
    }
}

/// Implementation of TtsConversationChunkBuilder for ElevenLabsTtsConversationBuilder
impl TtsConversationChunkBuilder for ElevenLabsTtsConversationBuilder {
    type Conversation = ElevenLabsTtsConversation;

    fn synthesize(self) -> impl Stream<Item = AudioChunk> + Send + Unpin {
        let conversation = ElevenLabsTtsConversation {
            speakers: self.speakers,
            language: self.language,
            engine: self.engine,
            chunk_processor: self.chunk_processor,
            stability: self.stability,
            similarity: self.similarity,
            style_exaggeration: self.style_exaggeration,
            speaker_boost: self.speaker_boost,
        };

        Box::pin(conversation.into_stream())
    }
}

/// TTS conversation implementation for ElevenLabs that uses the actual ElevenLabs engine
pub struct ElevenLabsTtsConversation {
    speakers: Vec<SpeakerData>,
    language: Option<Language>,
    engine: TtsEngine, // This is the actual ElevenLabs engine!
    chunk_processor: Option<
        Box<dyn FnMut(std::result::Result<AudioChunk, VoiceError>) -> AudioChunk + Send + 'static>,
    >,
    // Voice settings from the builder
    stability: Option<Stability>,
    similarity: Option<Similarity>,
    style_exaggeration: Option<StyleExaggeration>,
    speaker_boost: Option<SpeakerBoost>,
}

impl ElevenLabsTtsConversation {
    /// Build voice settings from fluent-voice configuration
    fn build_voice_settings(&self, speaker: &SpeakerData) -> Option<ElevenLabsVoiceSettings> {
        // Only create voice settings if we have configuration to apply
        if self.stability.is_some()
            || self.similarity.is_some()
            || self.style_exaggeration.is_some()
            || self.speaker_boost.is_some()
            || speaker.speed_modifier.is_some()
        {
            Some(ElevenLabsVoiceSettings {
                stability: self.stability.as_ref().map(|s| s.value()).unwrap_or(0.5),
                similarity_boost: self.similarity.as_ref().map(|s| s.value()).unwrap_or(0.75),
                style: self.style_exaggeration.as_ref().map(|s| s.value()),
                use_speaker_boost: self.speaker_boost.as_ref().map(|s| s.is_enabled()),
                speed: speaker.speed_modifier,
            })
        } else {
            None
        }
    }
}

impl TtsConversation for ElevenLabsTtsConversation {
    type AudioStream = Pin<Box<dyn Stream<Item = AudioChunk> + Send>>;

    fn into_stream(mut self) -> Self::AudioStream {
        Box::pin(async_stream::stream! {
            // Process each speaker using the ACTUAL ElevenLabs engine
            for speaker in &self.speakers {
                // Use proper voice handling - prefer voice_id, fallback to default voice
                let voice_id = speaker.voice_id.as_deref().unwrap_or("21m00Tcm4TlvDq8ikWAM"); // Rachel voice ID

                // Build TTS request with all configuration
                let mut tts_builder = self.engine
                    .tts()
                    .text(&speaker.text)
                    .voice(voice_id);

                // Apply language settings if available
                if let Some(ref language) = self.language.as_ref().or(speaker.language.as_ref()) {
                    // Convert BCP-47 to ISO 639-1 for ElevenLabs
                    let iso_code = language.code().split('-').next().unwrap_or("en");
                    tts_builder = tts_builder.language(iso_code);
                }

                // Apply voice settings from builder configuration
                if let Some(voice_settings) = self.build_voice_settings(&speaker) {
                    tts_builder = tts_builder.voice_settings(voice_settings);
                }

                // Generate audio using ElevenLabs engine
                match tts_builder.generate().await {
                    Ok(audio_output) => {
                        // Convert ElevenLabs AudioOutput to fluent-voice AudioChunk
                        let format = audio_output.format();
                        let audio_bytes = audio_output.bytes().to_vec();
                        let chunk = AudioChunk::with_metadata(
                            audio_bytes,
                            0, // Duration will be calculated by AudioChunk
                            0, // Start time
                            Some(speaker.id.clone()),
                            Some(speaker.text.clone()),
                            Some(match format {
                        crate::engine::AudioFormat::Mp3_44100_32 => fluent_voice_domain::AudioFormat::Mp3Khz44_32,
                        crate::engine::AudioFormat::Mp3_44100_64 => fluent_voice_domain::AudioFormat::Mp3Khz44_64,
                        crate::engine::AudioFormat::Mp3_44100_96 => fluent_voice_domain::AudioFormat::Mp3Khz44_96,
                        crate::engine::AudioFormat::Mp3_44100_128 => fluent_voice_domain::AudioFormat::Mp3Khz44_128,
                        crate::engine::AudioFormat::Mp3_44100_192 => fluent_voice_domain::AudioFormat::Mp3Khz44_192,
                        crate::engine::AudioFormat::Pcm16000 => fluent_voice_domain::AudioFormat::Pcm16Khz,
                        crate::engine::AudioFormat::Pcm22050 => fluent_voice_domain::AudioFormat::Pcm22Khz,
                        crate::engine::AudioFormat::Pcm24000 => fluent_voice_domain::AudioFormat::Pcm24Khz,
                        _ => fluent_voice_domain::AudioFormat::Mp3Khz44_128, // Default fallback
                    }),
                        );

                        // Apply chunk processor if available (CRITICAL FIX)
                        let final_chunk = if let Some(ref mut processor) = self.chunk_processor {
                            processor(Ok(chunk))
                        } else {
                            chunk
                        };

                        yield final_chunk;
                    },
                    Err(e) => {
                        // Create error chunk
                        let error_chunk = AudioChunk::bad_chunk(
                            format!("ElevenLabs TTS failed for speaker {}: {}", speaker.id, e)
                        );

                        // Apply chunk processor to error case (CRITICAL FIX)
                        let final_error_chunk = if let Some(ref mut processor) = self.chunk_processor {
                            processor(Err(VoiceError::Synthesis(format!("TTS failed: {}", e))))
                        } else {
                            error_chunk
                        };

                        yield final_error_chunk;
                    }
                }
            }
        })
    }
}
