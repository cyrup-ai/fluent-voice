//! Fluent-voice builder API for ElevenLabs
//!
//! This module provides the ONLY public API for the ElevenLabs crate.
//! All internal engine details are hidden - users must use these builders.

#![allow(dead_code)]

pub use crate::engine::{AudioOutput, AudioStream, Result, TranscriptOutput, TranscriptStream};
use crate::engine::{TtsEngine as InternalEngine, TtsEngineBuilder as InternalEngineBuilder};

/// Main entry point for fluent-voice API with ElevenLabs
pub struct FluentVoice;

impl FluentVoice {
    /// Create a new TTS (Text-to-Speech) builder chain
    pub fn tts() -> TtsBuilder {
        TtsBuilder::new()
    }

    /// Create a new STT (Speech-to-Text) builder chain
    pub fn stt() -> SttBuilder {
        SttBuilder::new()
    }

    /// Create a new Voice Changer (Speech-to-Speech) builder chain
    pub fn voice_changer() -> VoiceChangerBuilder {
        VoiceChangerBuilder::new()
    }
}

/// TTS builder for fluent-voice API
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
        let speaker_builder = SpeakerBuilder::new(engine);
        configure(speaker_builder)
    }
}

/// Speaker configuration builder
pub struct SpeakerBuilder {
    engine: InternalEngine,
}

impl SpeakerBuilder {
    fn new(engine: InternalEngine) -> Self {
        Self { engine }
    }

    /// Create a named speaker
    pub fn named(self, name: impl Into<String>) -> SpeakerConfig {
        SpeakerConfig {
            engine: self.engine,
            voice_name: name.into(),
            text: None,
        }
    }
}

/// Speaker configuration
pub struct SpeakerConfig {
    engine: InternalEngine,
    voice_name: String,
    text: Option<String>,
}

impl SpeakerConfig {
    /// Set the text for this speaker to say
    pub fn speak(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Build and add this speaker to the conversation
    pub fn build(self) -> ConversationBuilder {
        let mut tts = self.engine.tts();

        // Configure the speaker
        tts = tts.voice(self.voice_name);
        if let Some(text) = self.text {
            tts = tts.text(text);
        }

        ConversationBuilder { tts_builder: tts }
    }
}

/// Conversation builder for synthesis
pub struct ConversationBuilder {
    tts_builder: crate::engine::TtsBuilder,
}

impl ConversationBuilder {
    /// Synthesize the conversation
    pub async fn synthesize<F, R>(self, matcher: F) -> Result<R>
    where
        F: FnOnce(Result<AudioStream>) -> R,
    {
        let result = self.tts_builder.stream().await;
        let mapped_result = result.map_err(|e| e.into());
        Ok(matcher(mapped_result))
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

    /// Start microphone transcription
    pub fn with_microphone(self) -> Result<MicrophoneConfig> {
        let engine = self.engine_builder.build()?;
        let stt = engine.stt();
        Ok(MicrophoneConfig { stt_builder: stt })
    }
}

/// Transcription configuration
pub struct TranscriptionConfig {
    stt_builder: crate::engine::SttBuilder,
    path: String,
}

impl TranscriptionConfig {
    /// Set the model to use
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.stt_builder = self.stt_builder.model(model);
        self
    }

    /// Set the language
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.stt_builder = self.stt_builder.language(lang);
        self
    }

    /// Enable word timestamps
    pub fn with_word_timestamps(mut self) -> Self {
        self.stt_builder = self.stt_builder.with_word_timestamps();
        self
    }

    /// Enable speaker diarization
    pub fn diarization(mut self, enabled: bool) -> Self {
        self.stt_builder = self.stt_builder.diarization(enabled);
        self
    }

    /// Tag audio events
    pub fn tag_audio_events(mut self, enabled: bool) -> Self {
        self.stt_builder = self.stt_builder.tag_audio_events(enabled);
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

/// Microphone configuration
pub struct MicrophoneConfig {
    stt_builder: crate::engine::SttBuilder,
}

impl MicrophoneConfig {
    /// Set the model to use
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.stt_builder = self.stt_builder.model(model);
        self
    }

    /// Set the language
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.stt_builder = self.stt_builder.language(lang);
        self
    }

    /// Enable VAD (Voice Activity Detection)
    pub fn vad_enabled(self, _enabled: bool) -> Self {
        // ElevenLabs doesn't support live STT yet, this is a placeholder
        self
    }

    /// Listen for transcription
    pub async fn listen<F, R>(self, matcher: F) -> Result<R>
    where
        F: FnOnce(Result<TranscriptStream>) -> R,
    {
        // ElevenLabs doesn't support live STT yet
        let result = Err("Live microphone transcription not yet supported by ElevenLabs".into());
        Ok(matcher(result))
    }
}

/// Re-export Speaker for convenience
pub struct Speaker;

impl Speaker {
    /// Create a named speaker
    pub fn named(name: impl Into<String>) -> SpeakerSetup {
        SpeakerSetup {
            name: name.into(),
            text: None,
        }
    }
}

/// Speaker setup configuration
pub struct SpeakerSetup {
    name: String,
    text: Option<String>,
}

impl SpeakerSetup {
    /// Set the text for this speaker
    pub fn speak(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Build the speaker configuration
    pub fn build(self) -> Self {
        self
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
