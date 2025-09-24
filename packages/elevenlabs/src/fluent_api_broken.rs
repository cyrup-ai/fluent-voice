//! Fluent-voice builder API for ElevenLabs
//!
//! This module provides the ONLY public API for the ElevenLabs crate.
//! All internal engine details are hidden - users must use these builders.

#![allow(dead_code)]

pub use crate::engine::{AudioOutput, AudioStream, Result, TranscriptOutput, TranscriptStream};
use crate::engine::{TtsEngine as InternalEngine, TtsEngineBuilder as InternalEngineBuilder};

// Import fluent-voice STT traits and types
use fluent_voice_domain::{
    Diarization, Language, NoiseReduction, Punctuation, SpeechSource, TimestampsGranularity,
    TranscriptionSegmentImpl, VadMode, VoiceError, WordTimestamps,
};

// Import futures for stream implementations
use futures::stream;
use futures_util;

/// Main entry point for fluent-voice API with ElevenLabs
pub struct FluentVoice;

impl FluentVoice {
    /// Create a new TTS (Text-to-Speech) builder chain
    pub fn tts() -> TtsEntry {
        TtsEntry::new()
    }

    /// Create a new STT (Speech-to-Text) builder chain
    pub fn stt() -> SttEntry {
        SttEntry::new()
    }

    /// Create a new Voice Changer (Speech-to-Speech) builder chain
    pub fn voice_changer() -> VoiceChangerBuilder {
        VoiceChangerBuilder::new()
    }
}

/// TTS Entry that provides conversation() method
pub struct TtsEntry;

impl TtsEntry {
    pub fn new() -> Self {
        Self
    }

    pub fn conversation(self) -> TtsConversationBuilder {
        TtsConversationBuilder::new()
    }
}

/// STT Entry that provides conversation() method
pub struct SttEntry;

impl SttEntry {
    pub fn new() -> Self {
        Self
    }

    pub fn conversation(self) -> SttConversationBuilder {
        SttConversationBuilder::new()
    }

    pub fn with_source(self, src: SpeechSource) -> SttConversationBuilder {
        self.conversation().with_source(src)
    }
}

/// TTS Conversation Builder that supports fluent-voice patterns
pub struct TtsConversationBuilder {
    engine_builder: InternalEngineBuilder,
    speakers: Vec<SpeakerConfig>,
}

#[derive(Clone)]
pub struct SpeakerConfig {
    pub name: String,
    pub text: String,
    pub voice_id: Option<String>,
    pub speed: Option<f32>,
}

impl TtsConversationBuilder {
    pub fn new() -> Self {
        Self {
            engine_builder: InternalEngine::elevenlabs(),
            speakers: Vec::new(),
        }
    }

    pub fn with_speaker<S: Into<SpeakerConfig>>(mut self, speaker: S) -> Self {
        self.speakers.push(speaker.into());
        self
    }

    pub fn on_chunk<F>(self, _processor: F) -> TtsChunkBuilder<F>
    where
        F: FnMut(std::result::Result<AudioChunk, Box<dyn std::error::Error + Send + Sync>>) -> AudioChunk + Send + 'static,
    {
        TtsChunkBuilder {
            conversation_builder: self,
            chunk_processor: _processor,
        }
    }

    pub async fn synthesize<M, S>(self, matcher: M) -> Result<S>
    where
        M: FnOnce(std::result::Result<TtsConversation, Box<dyn std::error::Error + Send + Sync>>) -> S + Send + 'static,
        S: futures_util::Stream<Item = i16> + Send + Unpin + 'static,
    {
        // Build engine and configure TTS
        let engine = self.engine_builder.build()?;
        let mut tts_builder = engine.tts();

        // Configure with first speaker
        if let Some(speaker) = self.speakers.first() {
            tts_builder = tts_builder.voice(&speaker.name);
            tts_builder = tts_builder.text(&speaker.text);
        }

        // Create conversation
        let conversation = TtsConversation {
            audio_stream: tts_builder.stream().await?,
        };

        Ok(matcher(Ok(conversation)))
    }
}

/// TTS Chunk Builder for on_chunk processing
pub struct TtsChunkBuilder<F> {
    conversation_builder: TtsConversationBuilder,
    chunk_processor: F,
}

impl<F> TtsChunkBuilder<F>
where
    F: FnMut(Result<AudioChunk>) -> AudioChunk + Send + 'static,
{
    pub async fn synthesize<M, S>(self, matcher: M) -> Result<S>
    where
        M: FnOnce(std::result::Result<TtsConversation, Box<dyn std::error::Error + Send + Sync>>) -> S + Send + 'static,
        S: futures_util::Stream<Item = i16> + Send + Unpin + 'static,
    {
        self.conversation_builder.synthesize(matcher).await
    }
}

/// TTS Conversation that provides into_stream()
pub struct TtsConversation {
    audio_stream: AudioStream,
}

impl TtsConversation {
    pub fn into_stream(self) -> impl futures_util::Stream<Item = i16> + Send + Unpin {
        self.audio_stream
    }
}

/// STT Conversation Builder that supports fluent-voice patterns
pub struct SttConversationBuilder {
    source: Option<SpeechSource>,
    language: Option<Language>,
    word_timestamps: bool,
}

impl SttConversationBuilder {
    pub fn new() -> Self {
        Self {
            source: None,
            language: None,
            word_timestamps: false,
        }
    }

    pub fn with_source(mut self, src: SpeechSource) -> Self {
        self.source = Some(src);
        self
    }

    pub fn language_hint(mut self, lang: Language) -> Self {
        self.language = Some(lang);
        self
    }

    pub fn word_timestamps(mut self, enabled: WordTimestamps) -> Self {
        self.word_timestamps = matches!(enabled, WordTimestamps::On);
        self
    }

    pub fn on_chunk<F>(self, _processor: F) -> SttChunkBuilder<F>
    where
        F: FnMut(std::result::Result<TranscriptionSegmentImpl, VoiceError>) -> TranscriptionSegmentImpl + Send + 'static,
    {
        SttChunkBuilder {
            conversation_builder: self,
            chunk_processor: _processor,
        }
    }

    pub fn listen<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(std::result::Result<SttConversation, VoiceError>) -> S + Send + 'static,
        S: futures_util::Stream<Item = std::result::Result<TranscriptionSegmentImpl, VoiceError>> + Send + Unpin + 'static,
    {
        // ElevenLabs doesn't support STT, return error
        let error = VoiceError::Configuration(
            "ElevenLabs does not support speech-to-text. Use fluent-voice default STT instead.".to_string()
        );
        matcher(Err(error))
    }
}

/// STT Chunk Builder for on_chunk processing
pub struct SttChunkBuilder<F> {
    conversation_builder: SttConversationBuilder,
    chunk_processor: F,
}

impl<F> SttChunkBuilder<F>
where
    F: FnMut(Result<TranscriptionSegmentImpl, VoiceError>) -> TranscriptionSegmentImpl + Send + 'static,
{
    pub fn listen<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(std::result::Result<SttConversation, VoiceError>) -> S + Send + 'static,
        S: futures_util::Stream<Item = std::result::Result<TranscriptionSegmentImpl, VoiceError>> + Send + Unpin + 'static,
    {
        self.conversation_builder.listen(matcher)
    }
}

/// STT Conversation (placeholder since ElevenLabs doesn't support STT)
pub struct SttConversation;

impl SttConversation {
    pub fn into_stream(self) -> impl futures_util::Stream<Item = std::result::Result<TranscriptionSegmentImpl, VoiceError>> + Send + Unpin {
        futures_util::stream::empty()
    }
}

/// Simple AudioChunk for compatibility
#[derive(Clone)]
pub struct AudioChunk {
    data: Vec<u8>,
}

impl AudioChunk {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    pub fn bad_chunk(error: String) -> Self {
        Self {
            data: Vec::new(),
        }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

impl From<Vec<u8>> for AudioChunk {
    fn from(data: Vec<u8>) -> Self {
        Self::new(data)
    }
}

/// Speaker trait implementation for compatibility
pub trait Speaker {
    fn speaker(name: &str) -> SpeakerBuilder;
}

pub struct SpeakerBuilder {
    name: String,
    text: Option<String>,
    voice_id: Option<String>,
    speed: Option<f32>,
}

impl SpeakerBuilder {
    pub fn new(name: String) -> Self {
        Self {
            name,
            text: None,
            voice_id: None,
            speed: None,
        }
    }

    pub fn speak(mut self, text: &str) -> Self {
        self.text = Some(text.to_string());
        self
    }

    pub fn voice_id(mut self, id: VoiceId) -> Self {
        self.voice_id = Some(id.to_string());
        self
    }

    pub fn with_speed_modifier(mut self, speed: VocalSpeedMod) -> Self {
        self.speed = Some(speed.0);
        self
    }

    pub fn build(self) -> SpeakerConfig {
        SpeakerConfig {
            name: self.name,
            text: self.text.unwrap_or_default(),
            voice_id: self.voice_id,
            speed: self.speed,
        }
    }
}

impl Speaker for SpeakerBuilder {
    fn speaker(name: &str) -> SpeakerBuilder {
        SpeakerBuilder::new(name.to_string())
    }
}

impl From<SpeakerBuilder> for SpeakerConfig {
    fn from(builder: SpeakerBuilder) -> Self {
        builder.build()
    }
}

/// Voice ID wrapper
#[derive(Clone)]
pub struct VoiceId(String);

impl VoiceId {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
}

impl ToString for VoiceId {
    fn to_string(&self) -> String {
        self.0.clone()
    }
}

/// Vocal speed modifier
#[derive(Clone, Copy)]
pub struct VocalSpeedMod(pub f32);

/// Legacy TTS builder for backwards compatibility
pub struct TtsBuilder {
    engine_builder: InternalEngineBuilder,
}

impl TtsBuilder {
    fn new() -> Self {
        Self {
            engine_builder: InternalEngine::elevenlabs(),
        }
    }

    /// Set API key from environment variable
    pub fn api_key_from_env(mut self) -> Result<Self> {
        self.engine_builder = self.engine_builder.api_key_from_env()?;
        Ok(self)
    }

    /// Set API key directly
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.engine_builder = self.engine_builder.api_key(key);
        self
    }

    /// Enable HTTP/3 QUIC support
    pub fn http3_enabled(mut self, enabled: bool) -> Self {
        self.engine_builder = self.engine_builder.http3_enabled(enabled);
        self
    }

    /// Configure the TTS conversation
    pub fn with_speaker<F, R>(self, configure: F) -> Result<R>
    where
        F: FnOnce(SpeakerBuilder) -> Result<R>,
    {
        let engine = self.engine_builder.build()?;
        let speaker_builder = SpeakerBuilder::new("default".to_string());
        configure(speaker_builder)
    }
}

/// Speaker configuration builder
pub struct SpeakerConfigBuilder {
    engine: InternalEngine,
}

impl SpeakerConfigBuilder {
    fn new(engine: InternalEngine) -> Self {
        Self { engine }
    }

    /// Create a named speaker
    pub fn named(self, name: impl Into<String>) -> SpeakerConfig {
        SpeakerConfig {
            name: name.into(),
            text: String::new(),
            voice_id: None,
            speed: None,
        }
    }
}

/// STT builder for fluent-voice API
pub struct SttBuilder {
    engine_builder: InternalEngineBuilder,
}

impl SttBuilder {
    fn new() -> Self {
        Self {
            engine_builder: InternalEngine::elevenlabs(),
        }
    }

    /// Set API key from environment variable
    pub fn api_key_from_env(mut self) -> Result<Self> {
        self.engine_builder = self.engine_builder.api_key_from_env()?;
        Ok(self)
    }

    /// Set API key directly
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.engine_builder = self.engine_builder.api_key(key);
        self
    }

    /// Enable HTTP/3 QUIC support
    pub fn http3_enabled(mut self, enabled: bool) -> Self {
        self.engine_builder = self.engine_builder.http3_enabled(enabled);
        self
    }

    /// Transcribe an audio file
    pub fn transcribe(self, path: impl Into<String>) -> Result<TranscriptionConfig> {
        let engine = self.engine_builder.build()?;
        let stt = engine.stt();
        Ok(TranscriptionConfig {
            stt_builder: stt,
            path: path.into(),
        })
    }
}

/// Transcription configuration
pub struct TranscriptionConfig {
    stt_builder: crate::engine::SttBuilder,
    path: String,
}

impl TranscriptionConfig {
    /// Enable word timestamps
    pub fn with_word_timestamps(mut self) -> Self {
        self.stt_builder = self.stt_builder.with_word_timestamps();
        self
    }

    /// Set the language
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.stt_builder = self.stt_builder.language(lang);
        self
    }

    /// Emit the transcript
    pub async fn emit<F, R>(self, matcher: F) -> Result<R>
    where
        F: FnOnce(Result<TranscriptOutput>) -> R,
    {
        let result = self.stt_builder.transcribe(self.path).as_text().await;
        let mapped_result = result
            .map(|text| TranscriptOutput {
                text,
                language: String::new(),
                confidence: 1.0,
                words: Vec::new(),
            })
            .map_err(|e| e.into());
        Ok(matcher(mapped_result))
    }
}

/// Voice Changer builder for fluent-voice API (Speech-to-Speech conversion)
pub struct VoiceChangerBuilder {
    engine_builder: InternalEngineBuilder,
}

impl VoiceChangerBuilder {
    fn new() -> Self {
        Self {
            engine_builder: InternalEngine::elevenlabs(),
        }
    }

    /// Set API key from environment variable
    pub fn api_key_from_env(mut self) -> Result<Self> {
        self.engine_builder = self.engine_builder.api_key_from_env()?;
        Ok(self)
    }

    /// Set API key directly
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.engine_builder = self.engine_builder.api_key(key);
        self
    }

    /// Enable HTTP/3 QUIC support
    pub fn http3_enabled(mut self, enabled: bool) -> Self {
        self.engine_builder = self.engine_builder.http3_enabled(enabled);
        self
    }

    /// Configure voice conversion with target voice and source audio
    pub fn convert_to_voice(
        self,
        target_voice: impl Into<String>,
    ) -> Result<VoiceConversionConfig> {
        let engine = self.engine_builder.build()?;
        Ok(VoiceConversionConfig {
            engine,
            target_voice: target_voice.into(),
            source_audio: None,
        })
    }
}

/// Voice conversion configuration
pub struct VoiceConversionConfig {
    engine: InternalEngine,
    target_voice: String,
    source_audio: Option<String>,
}

impl VoiceConversionConfig {
    /// Set the source audio file to convert
    pub fn from_audio(mut self, audio_path: impl Into<String>) -> Self {
        self.source_audio = Some(audio_path.into());
        self
    }

    /// Convert the audio (non-streaming)
    pub async fn convert<F, R>(self, matcher: F) -> Result<R>
    where
        F: FnOnce(Result<AudioOutput>) -> R,
    {
        if let Some(audio_path) = self.source_audio {
            // Use the internal engine's voice_changer method
            let voice_changer = self.engine.voice_changer(&self.target_voice);
            let result = voice_changer.audio_file(audio_path).convert().await;
            Ok(matcher(result))
        } else {
            let result = Err("Source audio file is required for voice conversion".into());
            Ok(matcher(result))
        }
    }

    /// Convert the audio (streaming)
    pub async fn stream<F, R>(self, matcher: F) -> Result<R>
    where
        F: FnOnce(Result<AudioStream>) -> R,
    {
        if let Some(audio_path) = self.source_audio {
            // Use the internal engine's voice_changer method
            let voice_changer = self.engine.voice_changer(&self.target_voice);
            let result = voice_changer.audio_file(audio_path).stream().await;
            Ok(matcher(result))
        } else {
            let result = Err("Source audio file is required for voice conversion streaming".into());
            Ok(matcher(result))
        }
    }
}