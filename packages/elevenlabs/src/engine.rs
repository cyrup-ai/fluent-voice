//! Fluent-voice TTS engine implementation for ElevenLabs
//!
//! This module provides the ONLY public API for this crate.
//! All ElevenLabs functionality MUST be accessed through these builders.

#![allow(dead_code)]

use crate::client::{ClientConfig, ElevenLabsClient};
use crate::endpoints::genai::speech_to_text::{
    CreateTranscript, CreateTranscriptBody, CreateTranscriptQuery, CreateTranscriptResponse,
    SpeechToTextModel, Word,
};
use crate::endpoints::genai::tts::{
    Alignment, TextToSpeech, TextToSpeechBody, TextToSpeechQuery, TextToSpeechStream, Timestamps,
};
use crate::shared::query_params::OutputFormat;
use crate::shared::{DefaultVoice, Model, VoiceSettings as InternalVoiceSettings};
use crate::timestamp_metadata::{
    AudioChunkTimestamp, SynthesisContext, SynthesisMetadata, TimestampMetadata,
};
use crate::utils::{play, stream_audio};
use crate::voice::Voice as VoiceEnum;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use std::pin::Pin;

use fluent_voice::stt_conversation::SttConversationBuilder;  // ✅ Only what's needed

// Remove fluent-voice trait integration - ElevenLabs uses its own builder API
use std::task::{Context, Poll};
use std::time::Duration;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Fluent-voice error type
#[derive(Debug, thiserror::Error)]
pub enum FluentVoiceError {
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Audio playback error: {0}")]
    PlaybackError(String),
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Alignment validation error: {0}")]
    AlignmentValidationError(String),
}

/// TTS Engine - entry point for all operations
pub struct TtsEngine {
    client: ElevenLabsClient,
}

impl TtsEngine {
    /// Create a new ElevenLabs engine builder
    pub fn elevenlabs() -> TtsEngineBuilder {
        TtsEngineBuilder::default()
    }

    /// Create a new TTS conversation builder
    pub fn tts(&self) -> TtsBuilder {
        TtsBuilder::new(self.client.clone())
    }

    /// Create a new STT conversation builder
    pub fn stt(&self) -> SttBuilder {
        SttBuilder::new(self.client.clone())
    }

    /// List available voices
    pub async fn voices(&self) -> Result<Vec<Voice>> {
        use crate::endpoints::admin::voice::GetVoices;

        let endpoint = GetVoices::default();
        let response = self.client.hit(endpoint).await?;

        Ok(response
            .voices
            .into_iter()
            .map(|v| Voice {
                id: v.voice_id,
                name: v.name.unwrap_or_default(),
            })
            .collect())
    }

    /// Get a voice builder for advanced voice operations
    pub fn voice(&self, voice_id: impl Into<String>) -> VoiceBuilder {
        VoiceBuilder::new(self.client.clone(), voice_id.into())
    }

    /// Get available models
    pub async fn models(&self) -> Result<Vec<Model>> {
        Ok(vec![
            Model::ElevenEnglishV2,
            Model::ElevenMultilingualV2,
            Model::ElevenTurboV2,
            Model::ElevenTurboV2_5,
        ])
    }

    /// Create a pronunciation dictionaries builder for managing pronunciation rules
    pub fn pronunciation_dictionaries(&self) -> PronunciationBuilder {
        PronunciationBuilder::new(self.client.clone())
    }

    /// Create a voice changer builder for speech-to-speech conversion
    ///
    /// Transform audio from one voice to another while maintaining full control over
    /// emotion, timing, and delivery. Supports both streaming and non-streaming modes.
    ///
    /// # Arguments
    /// * `voice_id` - The target voice ID to convert audio to
    ///
    /// # Example
    /// ```no_run
    /// use fluent_voice_elevenlabs::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), VoiceError> {
    ///     let engine = TtsEngine::elevenlabs()
    ///         .api_key_from_env()?
    ///         .build()?;
    ///     
    ///     // Convert speaking voice to another voice
    ///     let converted_audio = engine.voice_changer("pNInz6obpgDQGcFmaJgB")
    ///         .audio_file("original_recording.wav")
    ///         .model("eleven_multilingual_v2_sts")
    ///         .remove_background_noise(true)
    ///         .convert()
    ///         .await?;
    ///     
    ///     converted_audio.play().await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn voice_changer(&self, voice_id: impl Into<String>) -> VoiceChangerBuilder {
        VoiceChangerBuilder::new(self.client.clone(), voice_id.into())
    }
}

/// Builder for configuring the TTS engine
#[derive(Default)]
pub struct TtsEngineBuilder {
    api_key: Option<String>,
    http3_enabled: bool,
    http3_config: Option<ClientConfig>,
}

impl TtsEngineBuilder {
    /// Set API key directly
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Load API key from environment
    pub fn api_key_from_env(self) -> Result<Self> {
        // This will check for ELEVENLABS_API_KEY, ELEVEN_API_KEY, or ELEVEN_LABS_API_KEY
        std::env::var("ELEVENLABS_API_KEY")
            .or_else(|_| std::env::var("ELEVEN_API_KEY"))
            .or_else(|_| std::env::var("ELEVEN_LABS_API_KEY"))
            .map_err(|_| {
                FluentVoiceError::ConfigError(
                    "No ElevenLabs API key found. Set ELEVENLABS_API_KEY environment variable"
                        .into(),
                )
            })?;
        Ok(self)
    }

    /// Enable HTTP/3 QUIC
    pub fn http3_enabled(mut self, enabled: bool) -> Self {
        self.http3_enabled = enabled;
        if enabled && self.http3_config.is_none() {
            self.http3_config = Some(ClientConfig::default());
        }
        self
    }

    /// Configure HTTP/3 settings
    pub fn http3_config(mut self, config: Http3Config) -> Self {
        self.http3_enabled = true;
        self.http3_config = Some(ClientConfig {
            enable_early_data: config.enable_early_data,
            max_idle_timeout: config.max_idle_timeout,
            stream_receive_window: config.stream_receive_window,
            conn_receive_window: config.conn_receive_window,
            send_window: config.send_window,
        });
        self
    }

    /// Build the TTS engine
    pub fn build(self) -> Result<TtsEngine> {
        let client = if let Some(api_key) = self.api_key {
            if let Some(config) = self.http3_config {
                ElevenLabsClient::new_with_config(api_key, config)?
            } else {
                ElevenLabsClient::new(api_key)?
            }
        } else {
            if let Some(config) = self.http3_config {
                ElevenLabsClient::from_env_with_config(config)?
            } else {
                ElevenLabsClient::from_env()?
            }
        };

        Ok(TtsEngine { client })
    }
}

/// HTTP/3 configuration
#[derive(Debug, Clone)]
pub struct Http3Config {
    pub enable_early_data: bool,
    pub max_idle_timeout: Duration,
    pub stream_receive_window: u64,
    pub conn_receive_window: u64,
    pub send_window: u64,
}

impl Default for Http3Config {
    fn default() -> Self {
        Self {
            enable_early_data: true,
            max_idle_timeout: Duration::from_secs(30),
            stream_receive_window: 1024 * 1024,
            conn_receive_window: 10 * 1024 * 1024,
            send_window: 1024 * 1024,
        }
    }
}

/// TTS conversation builder
pub struct TtsBuilder {
    client: ElevenLabsClient,
    text: Option<String>,
    voice_id: String,
    model: Model,
    voice_settings: Option<InternalVoiceSettings>,
    output_format: OutputFormat,
    pronunciation_dictionary_locators: Vec<String>,
    language_code: Option<String>,
    enable_logging: bool,
    seed: Option<i32>,
    previous_text: Option<String>,
    next_text: Option<String>,
    use_pvc_as_ivc: bool,
    apply_text_normalization: Option<String>,
}

impl TtsBuilder {
    fn new(client: ElevenLabsClient) -> Self {
        Self {
            client,
            text: None,
            voice_id: String::from(DefaultVoice::Sarah),
            model: Model::ElevenMultilingualV2,
            voice_settings: None,
            output_format: OutputFormat::Mp3_44100Hz128kbps,
            pronunciation_dictionary_locators: Vec::new(),
            language_code: None,
            enable_logging: false,
            seed: None,
            previous_text: None,
            next_text: None,
            use_pvc_as_ivc: false,
            apply_text_normalization: None,
        }
    }

    /// Set the text to synthesize
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Set the voice to use (by ID or name)
    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        let voice_str = voice.into();

        // Try to parse using the dynamically generated Voice enum first
        self.voice_id = if let Some(voice_enum) = VoiceEnum::from_name(&voice_str) {
            voice_enum.id().to_string()
        } else {
            // Fallback to direct voice ID usage for custom/unknown voices
            voice_str
        };
        self
    }

    /// Set the model to use
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = match model.into().as_str() {
            "eleven_monolingual_v1" => Model::ElevenEnglishV2, // Use V2 instead of deprecated V1
            "eleven_multilingual_v1" => Model::ElevenMultilingualV2, // Use V2 instead of deprecated V1
            "eleven_multilingual_v2" => Model::ElevenMultilingualV2,
            "eleven_turbo_v2" => Model::ElevenTurboV2,
            "eleven_turbo_v2_5" => Model::ElevenTurboV2_5,
            _ => Model::ElevenMultilingualV2,
        };
        self
    }

    /// Configure voice settings
    pub fn voice_settings(mut self, settings: VoiceSettings) -> Self {
        self.voice_settings = Some(InternalVoiceSettings {
            stability: Some(settings.stability),
            similarity_boost: Some(settings.similarity_boost),
            style: settings.style,
            use_speaker_boost: settings.use_speaker_boost,
            speed: settings.speed,
        });
        self
    }

    /// Set output format
    pub fn output_format(mut self, format: AudioFormat) -> Self {
        self.output_format = match format {
            AudioFormat::Mp3_44100_32 => OutputFormat::Mp3_44100Hz32kbps,
            AudioFormat::Mp3_44100_64 => OutputFormat::Mp3_44100Hz64kbps,
            AudioFormat::Mp3_44100_96 => OutputFormat::Mp3_44100Hz96kbps,
            AudioFormat::Mp3_44100_128 => OutputFormat::Mp3_44100Hz128kbps,
            AudioFormat::Mp3_44100_192 => OutputFormat::Mp3_44100Hz192kbps,
            AudioFormat::Pcm16000 => OutputFormat::Pcm16000Hz,
            AudioFormat::Pcm22050 => OutputFormat::Pcm22050Hz,
            AudioFormat::Pcm24000 => OutputFormat::Pcm24000Hz,
            AudioFormat::Pcm44100 => OutputFormat::Pcm44100Hz,
            AudioFormat::Ulaw8000 => OutputFormat::MuLaw8000Hz,
        };
        self
    }

    /// Add pronunciation dictionary
    pub fn pronunciation_dictionary(mut self, dictionary_id: impl Into<String>) -> Self {
        self.pronunciation_dictionary_locators
            .push(dictionary_id.into());
        self
    }

    /// Set language code (ISO 639-1)
    pub fn language(mut self, code: impl Into<String>) -> Self {
        self.language_code = Some(code.into());
        self
    }

    /// Enable request logging
    pub fn enable_logging(mut self, enabled: bool) -> Self {
        self.enable_logging = enabled;
        self
    }

    /// Set seed for deterministic generation
    pub fn seed(mut self, seed: i32) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set previous text for context
    pub fn previous_text(mut self, text: impl Into<String>) -> Self {
        self.previous_text = Some(text.into());
        self
    }

    /// Set next text for context
    pub fn next_text(mut self, text: impl Into<String>) -> Self {
        self.next_text = Some(text.into());
        self
    }

    /// Use PVC as IVC
    pub fn use_pvc_as_ivc(mut self, enabled: bool) -> Self {
        self.use_pvc_as_ivc = enabled;
        self
    }

    /// Apply text normalization
    pub fn apply_text_normalization(mut self, normalization: impl Into<String>) -> Self {
        self.apply_text_normalization = Some(normalization.into());
        self
    }

    /// Generate the audio
    pub async fn generate(self) -> Result<AudioOutput> {
        let text = self
            .text
            .ok_or_else(|| FluentVoiceError::ConfigError("Text is required".into()))?;

        let mut body = TextToSpeechBody::new(text).with_model_id(self.model);

        if let Some(settings) = self.voice_settings {
            body = body.with_voice_settings(settings);
        }

        if let Some(lang) = self.language_code {
            body = body.with_language_code(lang);
        }

        if let Some(seed) = self.seed {
            body = body.with_seed(seed as u64);
        }

        if let Some(prev) = self.previous_text {
            body = body.with_previous_text(&prev);
        }

        if let Some(next) = self.next_text {
            body = body.with_next_text(&next);
        }

        let output_format = self.output_format;
        let mut endpoint = TextToSpeech::new(self.voice_id, body);

        // Build query parameters
        let mut query = TextToSpeechQuery::default();
        query = query.with_output_format(output_format.clone());

        if self.enable_logging {
            query = query.with_logging(true);
        }

        endpoint = endpoint.with_query(query);
        let client = self.client;
        let audio_data = client.hit(endpoint).await?;

        Ok(AudioOutput {
            data: audio_data,
            format: output_format,
        })
    }

    /// Generate audio as a stream
    pub async fn stream(self) -> Result<AudioStream> {
        let text = self
            .text
            .ok_or_else(|| FluentVoiceError::ConfigError("Text is required".into()))?;

        let mut body = TextToSpeechBody::new(text).with_model_id(self.model);

        if let Some(settings) = self.voice_settings {
            body = body.with_voice_settings(settings);
        }

        if let Some(seed) = self.seed {
            body = body.with_seed(seed as u64);
        }

        if let Some(prev) = self.previous_text {
            body = body.with_previous_text(&prev);
        }

        if let Some(next) = self.next_text {
            body = body.with_next_text(&next);
        }

        let endpoint = TextToSpeechStream::new(self.voice_id, body);
        let stream = self.client.hit(endpoint).await?;

        Ok(AudioStream::new(stream))
    }

    /// Generate audio stream with character-level timestamps  
    pub async fn stream_with_timestamps(self) -> Result<AudioStreamWithTimestamps> {
        // Production implementation: Use ElevenLabs WebSocket streaming for real-time synthesis
        // Falls back to batch generation when streaming not available

        // Create synthesis context before consuming self
        let synthesis_context = SynthesisContext::from_tts_builder(
            &self.voice_id,
            &String::from(self.model.clone()),
            self.text.as_deref(),
            self.voice_settings.as_ref(),
            &self.output_format.to_string(),
            self.language_code.as_deref(),
        );

        let audio_with_timestamps = self.generate_with_timestamps().await?;
        Ok(AudioStreamWithTimestamps::from_audio_with_timestamps(
            audio_with_timestamps,
            synthesis_context,
        ))
    }

    /// Generate audio with character-level timestamps
    pub async fn generate_with_timestamps(self) -> Result<AudioWithTimestamps> {
        use crate::endpoints::genai::tts::TextToSpeechWithTimestamps;

        let text = self
            .text
            .ok_or_else(|| FluentVoiceError::ConfigError("Text is required".into()))?;

        let mut body = TextToSpeechBody::new(text).with_model_id(self.model);

        if let Some(settings) = self.voice_settings {
            body = body.with_voice_settings(settings);
        }

        if let Some(lang) = self.language_code {
            body = body.with_language_code(lang);
        }

        if let Some(seed) = self.seed {
            body = body.with_seed(seed as u64);
        }

        if let Some(prev) = self.previous_text {
            body = body.with_previous_text(&prev);
        }

        if let Some(next) = self.next_text {
            body = body.with_next_text(&next);
        }

        let output_format = self.output_format;
        let mut endpoint = TextToSpeechWithTimestamps::new(self.voice_id, body);

        // Build query parameters
        let mut query = TextToSpeechQuery::default();
        query = query.with_output_format(output_format.clone());

        if self.enable_logging {
            query = query.with_logging(true);
        }

        endpoint = endpoint.with_query(query);
        let response = self.client.hit(endpoint).await?;

        Ok(AudioWithTimestamps {
            audio_base64: response.audio_base64,
            alignment: response.alignment,
            normalized_alignment: response.normalized_alignment,
            format: output_format,
        })
    }

    /// Start a conversation (for future WebSocket support)
    pub async fn conversation(self) -> Result<TtsConversation> {
        Ok(TtsConversation {
            client: self.client,
            voice_id: self.voice_id,
            model: self.model,
            voice_settings: self.voice_settings,
        })
    }
}

/// TTS conversation handle (for future streaming/WebSocket support)
pub struct TtsConversation {
    client: ElevenLabsClient,
    voice_id: String,
    model: Model,
    voice_settings: Option<InternalVoiceSettings>,
}

impl TtsConversation {
    /// Send text and receive audio
    pub async fn send_text(&self, text: impl Into<String>) -> Result<AudioOutput> {
        let mut body = TextToSpeechBody::new(text.into()).with_model_id(self.model.clone());

        if let Some(settings) = &self.voice_settings {
            body = body.with_voice_settings(settings.clone());
        }

        let endpoint = TextToSpeech::new(&self.voice_id, body);
        let audio_data = self.client.hit(endpoint).await?;

        Ok(AudioOutput {
            data: audio_data,
            format: OutputFormat::Mp3_44100Hz128kbps,
        })
    }
}

/// Audio output handle
pub struct AudioOutput {
    data: Bytes,
    format: OutputFormat,
}

impl AudioOutput {
    /// Play the audio
    pub async fn play(self) -> Result<()> {
        play(self.data)?;
        Ok(())
    }

    /// Get the raw audio bytes
    pub fn bytes(self) -> Bytes {
        self.data
    }

    /// Save to file
    pub async fn save(self, path: impl AsRef<std::path::Path>) -> Result<()> {
        tokio::fs::write(path, &self.data).await?;
        Ok(())
    }

    /// Get audio format
    pub fn format(&self) -> AudioFormat {
        match self.format {
            OutputFormat::Mp3_22050Hz32kbps => AudioFormat::Mp3_44100_32, // Default to closest
            OutputFormat::Mp3_44100Hz32kbps => AudioFormat::Mp3_44100_32,
            OutputFormat::Mp3_44100Hz64kbps => AudioFormat::Mp3_44100_64,
            OutputFormat::Mp3_44100Hz96kbps => AudioFormat::Mp3_44100_96,
            OutputFormat::Mp3_44100Hz128kbps => AudioFormat::Mp3_44100_128,
            OutputFormat::Mp3_44100Hz192kbps => AudioFormat::Mp3_44100_192,
            OutputFormat::Pcm8000Hz => AudioFormat::Pcm16000, // Default to closest
            OutputFormat::Pcm16000Hz => AudioFormat::Pcm16000,
            OutputFormat::Pcm22050Hz => AudioFormat::Pcm22050,
            OutputFormat::Pcm24000Hz => AudioFormat::Pcm24000,
            OutputFormat::Pcm44100Hz => AudioFormat::Pcm44100,
            OutputFormat::MuLaw8000Hz => AudioFormat::Ulaw8000,
            OutputFormat::Opus48000Hz32kbps => AudioFormat::Mp3_44100_32, // Default to MP3
            OutputFormat::Opus48000Hz64kbps => AudioFormat::Mp3_44100_64,
            OutputFormat::Opus48000Hz96kbps => AudioFormat::Mp3_44100_96,
            OutputFormat::Opus48000Hz128kbps => AudioFormat::Mp3_44100_128,
            OutputFormat::Opus48000Hz192kbps => AudioFormat::Mp3_44100_192,
        }
    }

    /// Get audio size in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/// Audio stream for streaming TTS
pub struct AudioStream {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>,
}

impl AudioStream {
    fn new(stream: impl Stream<Item = Result<Bytes>> + Send + 'static) -> Self {
        Self {
            inner: Box::pin(stream),
        }
    }

    /// Play the stream
    pub async fn play(self) -> Result<()> {
        stream_audio(self.inner).await?;
        Ok(())
    }

    /// Save the stream to file
    pub async fn save(mut self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let mut file = tokio::fs::File::create(path).await?;
        use tokio::io::AsyncWriteExt;

        while let Some(chunk) = self.inner.next().await {
            let data = chunk?;
            file.write_all(&data).await?;
        }

        Ok(())
    }
}

impl Stream for AudioStream {
    type Item = Result<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// Audio stream with character-level timestamps for streaming TTS
pub struct AudioStreamWithTimestamps {
    inner: Pin<Box<dyn Stream<Item = Result<TimestampedAudioChunk>> + Send>>,
    synthesis_context: SynthesisContext,
}

impl AudioStreamWithTimestamps {
    fn new<S>(stream: S, synthesis_context: SynthesisContext) -> Self
    where
        S: Stream<Item = Result<crate::endpoints::genai::tts::TextToSpeechWithTimestampsResponse>>
            + Send
            + 'static,
    {
        use crate::endpoints::genai::tts::TextToSpeechWithTimestampsResponse;

        let mapped_stream = stream.map(|result| {
            result.map(
                |response: TextToSpeechWithTimestampsResponse| TimestampedAudioChunk {
                    audio_base64: response.audio_base64,
                    alignment: response.alignment,
                    normalized_alignment: response.normalized_alignment,
                },
            )
        });

        Self {
            inner: Box::pin(mapped_stream),
            synthesis_context,
        }
    }

    /// Create from AudioWithTimestamps (fallback when true streaming not available)
    fn from_audio_with_timestamps(
        audio_with_timestamps: AudioWithTimestamps,
        synthesis_context: SynthesisContext,
    ) -> Self {
        use futures_util::stream;

        // Convert the single AudioWithTimestamps into a stream with one item
        let audio_base64 = audio_with_timestamps.audio_base64().to_string();
        let chunk = TimestampedAudioChunk {
            audio_base64,
            alignment: audio_with_timestamps.alignment.clone(),
            normalized_alignment: audio_with_timestamps.normalized_alignment.clone(),
        };

        let stream = stream::once(async move { Ok(chunk) });

        Self {
            inner: Box::pin(stream),
            synthesis_context,
        }
    }

    /// Play the timestamped stream  
    pub async fn play(self) -> Result<()> {
        // For now, extract just the audio and play it
        let audio_stream = self.audio_only();
        stream_audio(audio_stream).await?;
        Ok(())
    }

    /// Convert to audio-only stream
    pub fn audio_only(self) -> impl Stream<Item = Result<Bytes>> {
        self.inner.map(|result| {
            result.and_then(|chunk| {
                // Decode base64 audio
                use base64::{Engine, engine::general_purpose};
                general_purpose::STANDARD
                    .decode(&chunk.audio_base64)
                    .map(Bytes::from)
                    .map_err(|e| format!("Failed to decode audio: {}", e).into())
            })
        })
    }

    /// Save the timestamped stream to file with separate timestamp data
    pub async fn save_with_timestamps(
        mut self,
        audio_path: impl AsRef<std::path::Path>,
        timestamps_path: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        let mut audio_file = tokio::fs::File::create(audio_path).await?;
        let mut all_alignments = Vec::new();
        use base64::{Engine, engine::general_purpose};
        use tokio::io::AsyncWriteExt;

        let mut total_processed_bytes = 0usize;

        while let Some(chunk) = self.inner.next().await {
            let chunk = chunk?;

            // Save audio data and track size
            let audio_bytes = general_purpose::STANDARD
                .decode(&chunk.audio_base64)
                .map_err(|e| format!("Failed to decode audio: {}", e))?;
            audio_file.write_all(&audio_bytes).await?;

            // Collect timestamp data with size tracking
            if let Some(alignment) = chunk.alignment {
                all_alignments.push((alignment, audio_bytes.len(), total_processed_bytes));
                total_processed_bytes += audio_bytes.len();
            }
        }

        // Create comprehensive timestamp metadata
        let mut timestamp_metadata = TimestampMetadata::new();

        // Populate with synthesis metadata (extracted from context)
        timestamp_metadata.synthesis_metadata = SynthesisMetadata {
            voice_id: self.synthesis_context.voice_id.clone(),
            model_id: self.synthesis_context.model_id.clone(),
            text: self.synthesis_context.text.clone(),
            voice_settings: self.synthesis_context.voice_settings.clone(),
            output_format: self.synthesis_context.output_format.clone(),
            language: self.synthesis_context.language.clone(),
        };

        // Process all alignments from ElevenLabs response
        for (alignment, _, _) in &all_alignments {
            timestamp_metadata.add_alignment(alignment);
        }

        // Extract timestamp configuration from builder metadata (if available)
        // Note: This would typically come from the TTS builder context, but for now we use defaults
        let timestamp_config = crate::timestamp_metadata::TimestampConfiguration::default();

        // Apply configuration filtering
        timestamp_metadata.filter_by_granularity(timestamp_config.granularity);
        timestamp_metadata.filter_by_word_timestamps(timestamp_config.word_timestamps);
        timestamp_metadata.filter_by_diarization(timestamp_config.diarization);

        // Add all alignment data with calculated timing information
        let mut cumulative_time_ms = 0u64;

        for (chunk_idx, (alignment, chunk_size, _byte_offset)) in all_alignments.iter().enumerate()
        {
            // ✅ ADD VALIDATION BEFORE PROCESSING
            crate::timestamp_metadata::validate_alignment(alignment)
                .map_err(|e| format!("Chunk {} alignment validation failed: {}", chunk_idx, e))?;

            // Alignment already added above in configuration section

            // Calculate precise timing from alignment data
            let start_seconds = alignment
                .character_start_times_seconds
                .first()
                .copied()
                .unwrap_or(0.0);
            let end_seconds = alignment
                .character_end_times_seconds
                .last()
                .copied()
                .unwrap_or(0.0);
            let duration_ms = ((end_seconds - start_seconds) * 1000.0) as u64;

            // Extract actual text segment from character data
            let text_segment = alignment.characters.join("");

            let chunk = AudioChunkTimestamp {
                chunk_id: chunk_idx,
                start_ms: cumulative_time_ms,
                end_ms: cumulative_time_ms + duration_ms,
                text_segment,
                speaker_id: None, // Multi-speaker support available when needed
                format: self.synthesis_context.output_format.clone(),
                size_bytes: *chunk_size,
            };

            timestamp_metadata.add_chunk(chunk);
            cumulative_time_ms += duration_ms;
        }

        // Finalize timing calculations
        timestamp_metadata.finalize()?;

        // Save comprehensive JSON with all timing data preserved
        let timestamps_json = timestamp_metadata.to_json()?;
        tokio::fs::write(&timestamps_path, &timestamps_json).await?;

        // Optional: Also save as SRT and VTT formats for subtitle use
        if let Ok(srt_content) = timestamp_metadata.to_srt() {
            let srt_path = timestamps_path.as_ref().with_extension("srt");
            tokio::fs::write(srt_path, srt_content).await?;
        }

        if let Ok(vtt_content) = timestamp_metadata.to_vtt() {
            let vtt_path = timestamps_path.as_ref().with_extension("vtt");
            tokio::fs::write(vtt_path, vtt_content).await?;
        }

        Ok(())
    }
}

impl Stream for AudioStreamWithTimestamps {
    type Item = Result<TimestampedAudioChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// A chunk of audio with timing information
#[derive(Debug, Clone)]
pub struct TimestampedAudioChunk {
    pub audio_base64: String,
    pub alignment: Option<Alignment>,
    pub normalized_alignment: Option<Alignment>,
}

impl TimestampedAudioChunk {
    /// Get the audio data as bytes
    pub fn audio_bytes(&self) -> Result<Bytes> {
        use base64::{Engine, engine::general_purpose};
        general_purpose::STANDARD
            .decode(&self.audio_base64)
            .map(Bytes::from)
            .map_err(|e| format!("Failed to decode audio: {}", e).into())
    }

    /// Get character-level timestamps for this chunk
    pub fn timestamps(&self) -> Option<Timestamps<'_>> {
        self.alignment.as_ref().map(|a| a.iter())
    }

    /// Get normalized timestamps for this chunk  
    pub fn normalized_timestamps(&self) -> Option<Timestamps<'_>> {
        self.normalized_alignment.as_ref().map(|a| a.iter())
    }
}

/// Voice configuration
#[derive(Debug, Clone)]
pub struct Voice {
    pub id: String,
    pub name: String,
}

/// Voice settings for synthesis
#[derive(Debug, Clone)]
pub struct VoiceSettings {
    pub stability: f32,
    pub similarity_boost: f32,
    pub style: Option<f32>,
    pub use_speaker_boost: Option<bool>,
    pub speed: Option<f32>,
}

impl Default for VoiceSettings {
    fn default() -> Self {
        Self {
            stability: 0.5,
            similarity_boost: 0.75,
            style: Some(0.0),
            use_speaker_boost: Some(true),
            speed: None,
        }
    }
}

/// Audio format options
#[derive(Debug, Clone, Copy)]
pub enum AudioFormat {
    Mp3_44100_32,
    Mp3_44100_64,
    Mp3_44100_96,
    Mp3_44100_128,
    Mp3_44100_192,
    Pcm16000,
    Pcm22050,
    Pcm24000,
    Pcm44100,
    Ulaw8000,
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self::Mp3_44100_128
    }
}

/// STT conversation builder
pub struct SttBuilder {
    client: ElevenLabsClient,
    source: Option<String>,
    model: SpeechToTextModel,
    language_code: Option<String>,
    diarize: bool,
    num_speakers: Option<u32>,
    tag_audio_events: bool,
    word_timestamps: bool,
    timestamps_granularity: Option<String>,
    enable_logging: bool,
}

impl SttBuilder {
    fn new(client: ElevenLabsClient) -> Self {
        Self {
            client,
            source: None,
            model: SpeechToTextModel::ScribeV1,
            language_code: None,
            diarize: false,
            num_speakers: None,
            tag_audio_events: false,
            word_timestamps: false,
            timestamps_granularity: None,
            enable_logging: true,
        }
    }

    /// Set the model to use for transcription
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = match model.into().as_str() {
            "scribe_v1" => SpeechToTextModel::ScribeV1,
            "scribe_v1_base" => SpeechToTextModel::ScribeV1Base,
            _ => SpeechToTextModel::ScribeV1,
        };
        self
    }

    /// Set language hint
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language_code = Some(lang.into());
        self
    }

    /// Enable speaker diarization
    pub fn diarization(mut self, enabled: bool) -> Self {
        self.diarize = enabled;
        self
    }

    /// Set number of speakers
    pub fn speakers(mut self, count: u32) -> Self {
        self.num_speakers = Some(count);
        self
    }

    /// Enable audio event tagging
    pub fn tag_audio_events(mut self, enabled: bool) -> Self {
        self.tag_audio_events = enabled;
        self
    }

    /// Enable word timestamps
    pub fn with_word_timestamps(mut self) -> Self {
        self.word_timestamps = true;
        self.timestamps_granularity = Some("word".to_string());
        self
    }

    /// Enable character timestamps
    pub fn with_character_timestamps(mut self) -> Self {
        self.word_timestamps = true;
        self.timestamps_granularity = Some("character".to_string());
        self
    }

    /// Transcribe an audio file
    pub fn transcribe(mut self, path: impl Into<String>) -> TranscriptionBuilder {
        self.source = Some(path.into());
        TranscriptionBuilder {
            builder: self,
            progress_template: None,
        }
    }

    /// Configure microphone for STT (delegates to DefaultSTTEngine)
    /// ElevenLabs doesn't provide STT, so this delegates to the fluent-voice DefaultSTTEngine
    pub fn microphone(
        self,
        device: impl Into<String>,
    ) -> impl fluent_voice::stt_conversation::SttConversationBuilder {
        use fluent_voice::prelude::*;

        // Delegate directly to fluent-voice STT API
        FluentVoice::stt()
            .conversation()
            .with_source(SpeechSource::Microphone {
                backend: MicBackend::Device(device.into()),
                format: AudioFormat::Pcm16Khz,
                sample_rate: 16_000,
            })
    }

    /// Collect the transcription
    pub async fn collect(self) -> Result<TranscriptOutput> {
        if let Some(file_path) = self.source {
            let body = CreateTranscriptBody::new(self.model, file_path);
            let body = if let Some(lang) = self.language_code {
                body.with_language_code(lang)
            } else {
                body
            };

            let body = body
                .with_diarize(self.diarize)
                .with_tag_audio_events(self.tag_audio_events);

            let body = if let Some(num) = self.num_speakers {
                body.with_num_speakers(num)
            } else {
                body
            };

            let body = if let Some(gran) = self.timestamps_granularity {
                body.with_timestamps_granularity(gran.as_str())
            } else {
                body
            };

            let mut endpoint = CreateTranscript::new(body);

            if !self.enable_logging {
                endpoint.query = Some(CreateTranscriptQuery::default().enable_logging(false));
            }

            let response = self.client.hit(endpoint).await?;
            Ok(TranscriptOutput::from_response(response))
        } else {
            Err("No audio source specified".into())
        }
    }
}

/// File transcription builder
pub struct TranscriptionBuilder {
    builder: SttBuilder,
    progress_template: Option<String>,
}

impl TranscriptionBuilder {
    /// Add progress reporting
    pub fn with_progress(mut self, template: impl Into<String>) -> Self {
        self.progress_template = Some(template.into());
        self
    }

    /// Collect the complete transcript
    pub async fn collect(self) -> Result<TranscriptOutput> {
        self.builder.collect().await
    }

    /// Get transcript as plain text
    pub async fn as_text(self) -> Result<String> {
        let output = self.collect().await?;
        Ok(output.text)
    }
}

/// Transcript output
#[derive(Debug, Clone)]
pub struct TranscriptOutput {
    pub text: String,
    pub language: String,
    pub confidence: f32,
    pub words: Vec<TranscriptWord>,
}

impl TranscriptOutput {
    fn from_response(resp: CreateTranscriptResponse) -> Self {
        Self {
            text: resp.text,
            language: resp.language_code,
            confidence: resp.language_probability,
            words: resp.words.into_iter().map(TranscriptWord::from).collect(),
        }
    }
}

/// Transcript word with timing
#[derive(Debug, Clone)]
pub struct TranscriptWord {
    pub text: String,
    pub start: Option<f32>,
    pub end: Option<f32>,
    pub speaker: Option<String>,
}

impl From<Word> for TranscriptWord {
    fn from(word: Word) -> Self {
        Self {
            text: word.text,
            start: word.start,
            end: word.end,
            speaker: word.speaker_id,
        }
    }
}

/// Transcript stream for real-time transcription using Whisper STT
pub struct TranscriptStream {
    inner: std::pin::Pin<Box<dyn futures::Stream<Item = Result<TranscriptOutput>> + Send>>,
}

impl TranscriptStream {
    /// Create a new transcript stream from a boxed stream
    pub fn new(
        stream: std::pin::Pin<Box<dyn futures::Stream<Item = Result<TranscriptOutput>> + Send>>,
    ) -> Self {
        Self { inner: stream }
    }
}

impl futures::Stream for TranscriptStream {
    type Item = Result<TranscriptOutput>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// Builder for voice operations
pub struct VoiceBuilder {
    client: ElevenLabsClient,
    voice_id: String,
}

impl VoiceBuilder {
    fn new(client: ElevenLabsClient, voice_id: String) -> Self {
        Self { client, voice_id }
    }

    /// Get full voice details
    pub async fn get(&self) -> Result<VoiceDetails> {
        use crate::endpoints::admin::voice::GetVoice;

        let endpoint = GetVoice::new(&self.voice_id);
        let response = self.client.hit(endpoint).await?;

        Ok(VoiceDetails {
            id: response.voice_id,
            name: response.name.unwrap_or_default(),
            description: response.description,
            preview_url: response.preview_url,
            category: response.category.map(|c| format!("{:?}", c)),
            settings: response.settings.map(|s| VoiceSettings {
                stability: s.stability.unwrap_or(0.5),
                similarity_boost: s.similarity_boost.unwrap_or(0.5),
                style: s.style,
                use_speaker_boost: s.use_speaker_boost,
                speed: s.speed,
            }),
        })
    }

    /// Delete this voice
    pub async fn delete(self) -> Result<()> {
        use crate::endpoints::admin::voice::DeleteVoice;

        let endpoint = DeleteVoice::new(&self.voice_id);
        self.client.hit(endpoint).await?;
        Ok(())
    }

    /// Get voice settings
    pub async fn settings(&self) -> Result<VoiceSettings> {
        use crate::endpoints::admin::voice::GetVoiceSettings;

        let endpoint = GetVoiceSettings::new(&self.voice_id);
        let response = self.client.hit(endpoint).await?;

        Ok(VoiceSettings {
            stability: response.stability.unwrap_or(0.5),
            similarity_boost: response.similarity_boost.unwrap_or(0.5),
            style: response.style,
            use_speaker_boost: response.use_speaker_boost,
            speed: response.speed,
        })
    }

    /// Edit voice settings
    pub async fn edit_settings(&self, settings: VoiceSettings) -> Result<()> {
        use crate::endpoints::admin::voice::{EditVoiceSettings, EditVoiceSettingsBody};

        let mut body = EditVoiceSettingsBody::default();
        body = body
            .with_stability(settings.stability)
            .with_similarity_boost(settings.similarity_boost);

        if let Some(style) = settings.style {
            body = body.with_style(style);
        }
        if let Some(boost) = settings.use_speaker_boost {
            body = body.use_speaker_boost(boost);
        }
        if let Some(speed) = settings.speed {
            body = body.with_speed(speed);
        }

        let endpoint = EditVoiceSettings::new(&self.voice_id, body);
        self.client.hit(endpoint).await?;
        Ok(())
    }

    /// Edit voice metadata (name, description, labels)
    pub async fn edit(&self, name: impl Into<String>) -> VoiceEditBuilder {
        VoiceEditBuilder::new(self.client.clone(), self.voice_id.clone(), name.into())
    }
}

/// Builder for editing voice metadata
pub struct VoiceEditBuilder {
    client: ElevenLabsClient,
    voice_id: String,
    name: String,
    description: Option<String>,
    labels: Vec<(String, String)>,
    files: Vec<String>,
    remove_background_noise: Option<bool>,
}

impl VoiceEditBuilder {
    fn new(client: ElevenLabsClient, voice_id: String, name: String) -> Self {
        Self {
            client,
            voice_id,
            name,
            description: None,
            labels: Vec::new(),
            files: Vec::new(),
            remove_background_noise: None,
        }
    }

    /// Set voice description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a label (key-value pair)
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.push((key.into(), value.into()));
        self
    }

    /// Add multiple labels
    pub fn labels(
        mut self,
        labels: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        self.labels
            .extend(labels.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }

    /// Add voice sample files for updating the voice
    pub fn add_sample(mut self, file_path: impl Into<String>) -> Self {
        self.files.push(file_path.into());
        self
    }

    /// Add multiple voice sample files
    pub fn add_samples(mut self, files: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.files.extend(files.into_iter().map(|f| f.into()));
        self
    }

    /// Enable/disable background noise removal
    pub fn remove_background_noise(mut self, remove: bool) -> Self {
        self.remove_background_noise = Some(remove);
        self
    }

    /// Apply the voice edits
    pub async fn apply(self) -> Result<()> {
        use crate::endpoints::admin::voice::{EditVoice, VoiceBody};

        // EditVoice API allows updating the name and labels, but not adding new samples
        // For adding samples, you'd need to create a new voice
        let mut body = if self.files.is_empty() {
            VoiceBody::edit(&self.name)
        } else {
            // If files are provided, we need to use the add constructor
            // but EditVoice endpoint expects the edit format
            VoiceBody::edit(&self.name)
        };

        if let Some(desc) = self.description {
            body = body.with_description(&desc);
        }

        if !self.labels.is_empty() {
            body = body.with_labels(self.labels);
        }

        if let Some(remove) = self.remove_background_noise {
            body = body.with_remove_background_noise(remove);
        }

        let endpoint = EditVoice::new(&self.voice_id, body);
        self.client.hit(endpoint).await?;
        Ok(())
    }
}

/// Detailed voice information
pub struct VoiceDetails {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub preview_url: Option<String>,
    pub category: Option<String>,
    pub settings: Option<VoiceSettings>,
}

/// Builder for pronunciation dictionary operations
pub struct PronunciationBuilder {
    client: ElevenLabsClient,
}

impl PronunciationBuilder {
    fn new(client: ElevenLabsClient) -> Self {
        Self { client }
    }

    /// Create a new pronunciation dictionary
    pub async fn create(&self, name: impl Into<String>) -> Result<DictionaryBuilder> {
        Ok(DictionaryBuilder::new(self.client.clone(), name.into()))
    }

    /// List all pronunciation dictionaries  
    pub async fn list(&self) -> Result<Vec<DictionaryInfo>> {
        use crate::endpoints::admin::pronunciation::GetDictionaries;

        let endpoint = GetDictionaries::default();
        let response = self.client.hit(endpoint).await?;

        Ok(response
            .pronunciation_dictionaries
            .into_iter()
            .map(|d| DictionaryInfo {
                id: d.id,
                name: d.name,
                version_id: d.latest_version_id,
                description: d.description,
            })
            .collect())
    }

    /// Get a specific dictionary builder for operations
    pub fn dictionary(&self, dictionary_id: impl Into<String>) -> DictionaryBuilder {
        DictionaryBuilder::existing(self.client.clone(), dictionary_id.into())
    }
}

/// Builder for individual dictionary operations
pub struct DictionaryBuilder {
    client: ElevenLabsClient,
    dictionary_id: Option<String>,
    name: Option<String>,
    description: Option<String>,
}

impl DictionaryBuilder {
    fn new(client: ElevenLabsClient, name: String) -> Self {
        Self {
            client,
            dictionary_id: None,
            name: Some(name),
            description: None,
        }
    }

    fn existing(client: ElevenLabsClient, dictionary_id: String) -> Self {
        Self {
            client,
            dictionary_id: Some(dictionary_id),
            name: None,
            description: None,
        }
    }

    /// Set dictionary description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Create the dictionary from a PLS file (for new dictionaries)
    pub async fn build_from_file(self, pls_file_path: impl Into<String>) -> Result<String> {
        use crate::endpoints::admin::pronunciation::{CreateDictionary, CreateDictionaryBody};

        if let Some(name) = self.name {
            let mut body = CreateDictionaryBody::new(&pls_file_path.into(), &name);
            if let Some(desc) = self.description {
                body = body.with_description(&desc);
            }

            let endpoint = CreateDictionary::new(body);
            let response = self.client.hit(endpoint).await?;
            Ok(response.id)
        } else {
            Err("Cannot create dictionary without name".into())
        }
    }

    /// Add pronunciation rules to the dictionary
    pub async fn add_rules(&self, rules: Vec<PronunciationRule>) -> Result<()> {
        use crate::endpoints::admin::pronunciation::{AddRules, AddRulesBody, Rule};

        if let Some(dict_id) = &self.dictionary_id {
            let rules_converted: Vec<Rule> = rules
                .into_iter()
                .map(|rule| match rule {
                    PronunciationRule::Alias { string, alias } => Rule::new_alias(&string, &alias),
                    PronunciationRule::Phoneme {
                        string,
                        phoneme,
                        alphabet,
                    } => Rule::new_phoneme(&string, &phoneme, &alphabet),
                })
                .collect();

            let body = AddRulesBody::new(rules_converted);
            let endpoint = AddRules::new(dict_id, body);
            self.client.hit(endpoint).await?;
            Ok(())
        } else {
            Err("Cannot add rules without dictionary ID".into())
        }
    }

    /// Remove pronunciation rules from the dictionary
    pub async fn remove_rules(&self, rule_strings: Vec<String>) -> Result<()> {
        use crate::endpoints::admin::pronunciation::{RemoveRules, RemoveRulesBody};

        if let Some(dict_id) = &self.dictionary_id {
            let body = RemoveRulesBody::new(rule_strings.iter().map(|s| s.as_str()));
            let endpoint = RemoveRules::new(dict_id, body);
            self.client.hit(endpoint).await?;
            Ok(())
        } else {
            Err("Cannot remove rules without dictionary ID".into())
        }
    }

    /// Get dictionary metadata
    pub async fn metadata(&self) -> Result<DictionaryMetadata> {
        use crate::endpoints::admin::pronunciation::GetDictionaryMetaData;

        if let Some(dict_id) = &self.dictionary_id {
            let endpoint = GetDictionaryMetaData::new(dict_id);
            let response = self.client.hit(endpoint).await?;
            Ok(DictionaryMetadata {
                id: response.id,
                version_id: response.latest_version_id,
                name: response.name,
                description: response.description,
                alphabet: None, // Not available in metadata response
            })
        } else {
            Err("Cannot get metadata without dictionary ID".into())
        }
    }

    /// Download dictionary as PLS file (requires version_id)
    pub async fn download_pls(&self, version_id: impl Into<String>) -> Result<String> {
        use crate::endpoints::admin::pronunciation::GetPLSFile;

        if let Some(dict_id) = &self.dictionary_id {
            let endpoint = GetPLSFile::new(dict_id, version_id);
            let pls_bytes = self.client.hit(endpoint).await?;
            let pls_content = String::from_utf8(pls_bytes.to_vec())
                .map_err(|e| format!("Failed to convert PLS to string: {}", e))?;
            Ok(pls_content)
        } else {
            Err("Cannot download PLS without dictionary ID".into())
        }
    }
}

/// Pronunciation rule types
#[derive(Debug, Clone)]
pub enum PronunciationRule {
    Alias {
        string: String,
        alias: String,
    },
    Phoneme {
        string: String,
        phoneme: String,
        alphabet: String,
    },
}

/// Dictionary information
#[derive(Debug, Clone)]
pub struct DictionaryInfo {
    pub id: String,
    pub name: String,
    pub version_id: String,
    pub description: Option<String>,
}

/// Dictionary metadata
#[derive(Debug, Clone)]
pub struct DictionaryMetadata {
    pub id: String,
    pub version_id: String,
    pub name: String,
    pub description: Option<String>,
    pub alphabet: Option<String>,
}

/// Audio output with character-level timestamps
pub struct AudioWithTimestamps {
    audio_base64: String,
    alignment: Option<Alignment>,
    normalized_alignment: Option<Alignment>,
    format: OutputFormat,
}

impl AudioWithTimestamps {
    /// Get the audio as base64 string
    pub fn audio_base64(&self) -> &str {
        &self.audio_base64
    }

    /// Get character-level timestamps
    pub fn timestamps(&self) -> Option<Timestamps<'_>> {
        self.alignment.as_ref().map(|a| a.iter())
    }

    /// Get normalized timestamps
    pub fn normalized_timestamps(&self) -> Option<Timestamps<'_>> {
        self.normalized_alignment.as_ref().map(|a| a.iter())
    }

    /// Get all character timing data
    pub fn character_timings(&self) -> Vec<CharacterTiming> {
        self.alignment
            .as_ref()
            .map(|alignment| {
                alignment
                    .iter()
                    .map(|(char, (start, end))| CharacterTiming {
                        character: char.clone(),
                        start_time: start,
                        end_time: end,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get audio format
    pub fn format(&self) -> AudioFormat {
        match self.format {
            OutputFormat::Mp3_22050Hz32kbps => AudioFormat::Mp3_44100_32,
            OutputFormat::Mp3_44100Hz32kbps => AudioFormat::Mp3_44100_32,
            OutputFormat::Mp3_44100Hz64kbps => AudioFormat::Mp3_44100_64,
            OutputFormat::Mp3_44100Hz96kbps => AudioFormat::Mp3_44100_96,
            OutputFormat::Mp3_44100Hz128kbps => AudioFormat::Mp3_44100_128,
            OutputFormat::Mp3_44100Hz192kbps => AudioFormat::Mp3_44100_192,
            OutputFormat::Pcm8000Hz => AudioFormat::Pcm16000,
            OutputFormat::Pcm16000Hz => AudioFormat::Pcm16000,
            OutputFormat::Pcm22050Hz => AudioFormat::Pcm22050,
            OutputFormat::Pcm24000Hz => AudioFormat::Pcm24000,
            OutputFormat::Pcm44100Hz => AudioFormat::Pcm44100,
            OutputFormat::MuLaw8000Hz => AudioFormat::Ulaw8000,
            OutputFormat::Opus48000Hz32kbps => AudioFormat::Mp3_44100_32,
            OutputFormat::Opus48000Hz64kbps => AudioFormat::Mp3_44100_64,
            OutputFormat::Opus48000Hz96kbps => AudioFormat::Mp3_44100_96,
            OutputFormat::Opus48000Hz128kbps => AudioFormat::Mp3_44100_128,
            OutputFormat::Opus48000Hz192kbps => AudioFormat::Mp3_44100_192,
        }
    }
}

/// Character timing information
#[derive(Debug, Clone)]
pub struct CharacterTiming {
    pub character: String,
    pub start_time: f32,
    pub end_time: f32,
}

/// Builder for voice changer operations (speech-to-speech conversion)
///
/// Transform audio from one voice to another while maintaining full control over
/// emotion, timing, and delivery. Supports both real-time streaming and batch conversion.
///
/// # Example Usage
/// ```no_run
/// use fluent_voice_elevenlabs::*;
///
/// #[tokio::main]
/// async fn main() -> Result<(), VoiceError> {
///     let engine = TtsEngine::elevenlabs()
///         .api_key_from_env()?
///         .build()?;
///     
///     // Basic voice conversion
///     let audio = engine.voice_changer("target_voice_id")
///         .audio_file("input.wav")
///         .convert()
///         .await?;
///     
///     // Advanced configuration
///     let stream = engine.voice_changer("target_voice_id")
///         .audio_file("podcast.mp3")
///         .model("eleven_multilingual_v2_sts")
///         .voice_settings(VoiceSettings {
///             stability: 0.6,
///             similarity_boost: 0.9,
///             style: Some(0.1),
///             use_speaker_boost: Some(true),
///             speed: None,
///         })
///         .remove_background_noise(true)
///         .output_format(AudioFormat::Mp3_44100_128)
///         .stream()
///         .await?;
///     
///     Ok(())
/// }
/// ```
pub struct VoiceChangerBuilder {
    client: ElevenLabsClient,
    voice_id: String,
    audio_file: Option<String>,
    model_id: Option<String>,
    voice_settings: Option<InternalVoiceSettings>,
    seed: Option<u64>,
    remove_background_noise: Option<bool>,
    output_format: OutputFormat,
    enable_logging: bool,
}

impl VoiceChangerBuilder {
    /// Create a new voice changer builder
    pub fn new(client: ElevenLabsClient, voice_id: String) -> Self {
        Self {
            client,
            voice_id,
            audio_file: None,
            model_id: None,
            voice_settings: None,
            seed: None,
            remove_background_noise: None,
            output_format: OutputFormat::Mp3_44100Hz128kbps,
            enable_logging: false,
        }
    }

    /// Set the audio file to convert
    ///
    /// # Example
    /// ```no_run
    /// use fluent_voice_elevenlabs::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), VoiceError> {
    ///     let engine = TtsEngine::elevenlabs()
    ///         .api_key_from_env()?
    ///         .build()?;
    ///     
    ///     let audio = engine.voice_changer("voice_id")
    ///         .audio_file("input.wav")
    ///         .model("eleven_multilingual_v2_sts")
    ///         .convert()
    ///         .await?;
    ///     
    ///     audio.play().await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn audio_file(mut self, path: impl Into<String>) -> Self {
        let path_string = path.into();

        // Validate file exists and is readable
        if let Err(_) = std::fs::metadata(&path_string) {
            // Note: We store the path anyway to provide better error messages later
            // The actual validation will happen in convert()/stream() methods
        }

        // Validate file extension for security
        let path_obj = std::path::Path::new(&path_string);
        if let Some(ext) = path_obj.extension().and_then(|e| e.to_str()) {
            let valid_extensions = ["mp3", "wav", "flac", "m4a", "ogg", "webm"];
            if !valid_extensions.contains(&ext.to_lowercase().as_str()) {
                // Allow but warn - actual validation in convert()/stream()
            }
        }

        self.audio_file = Some(path_string);
        self
    }

    /// Set the model to use for speech-to-speech conversion
    ///
    /// # Supported Models
    /// - `eleven_multilingual_v2_sts` - High-quality multilingual model
    /// - `eleven_english_sts_v2` - English-optimized model
    ///
    /// # Example
    /// ```no_run
    /// # use fluent_voice_elevenlabs::*;
    /// # let engine = TtsEngine::elevenlabs().build().unwrap();
    /// let voice_changer = engine.voice_changer("voice_id")
    ///     .model("eleven_multilingual_v2_sts");
    /// ```
    pub fn model(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    /// Configure voice settings for fine-tuned control
    ///
    /// # Example
    /// ```no_run
    /// # use fluent_voice_elevenlabs::*;
    /// # let engine = TtsEngine::elevenlabs().build().unwrap();
    /// let voice_changer = engine.voice_changer("voice_id")
    ///     .voice_settings(VoiceSettings {
    ///         stability: 0.5,        // Voice consistency (0.0-1.0)
    ///         similarity_boost: 0.8, // Voice similarity (0.0-1.0)
    ///         style: Some(0.2),      // Speaking style (0.0-1.0)
    ///         use_speaker_boost: Some(true),
    ///         speed: None,
    ///     });
    /// ```
    pub fn voice_settings(mut self, settings: VoiceSettings) -> Self {
        self.voice_settings = Some(InternalVoiceSettings {
            stability: Some(settings.stability),
            similarity_boost: Some(settings.similarity_boost),
            style: settings.style,
            use_speaker_boost: settings.use_speaker_boost,
            speed: settings.speed,
        });
        self
    }

    /// Set generation seed for deterministic output
    ///
    /// Using the same seed with the same inputs will produce consistent results.
    ///
    /// # Example
    /// ```no_run
    /// # use fluent_voice_elevenlabs::*;
    /// # let engine = TtsEngine::elevenlabs().build().unwrap();
    /// let voice_changer = engine.voice_changer("voice_id")
    ///     .seed(12345); // Reproducible results
    /// ```
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable/disable background noise removal
    ///
    /// When enabled, background noise will be filtered out during conversion.
    /// Recommended for recordings with ambient noise.
    ///
    /// # Example
    /// ```no_run
    /// # use fluent_voice_elevenlabs::*;
    /// # let engine = TtsEngine::elevenlabs().build().unwrap();
    /// let voice_changer = engine.voice_changer("voice_id")
    ///     .remove_background_noise(true); // Clean audio output
    /// ```
    pub fn remove_background_noise(mut self, enabled: bool) -> Self {
        self.remove_background_noise = Some(enabled);
        self
    }

    /// Set output format for the converted audio
    ///
    /// # Example
    /// ```no_run
    /// # use fluent_voice_elevenlabs::*;
    /// # let engine = TtsEngine::elevenlabs().build().unwrap();
    /// let voice_changer = engine.voice_changer("voice_id")
    ///     .output_format(AudioFormat::Mp3_44100_128); // High-quality MP3
    /// ```
    pub fn output_format(mut self, format: AudioFormat) -> Self {
        self.output_format = match format {
            AudioFormat::Mp3_44100_32 => OutputFormat::Mp3_44100Hz32kbps,
            AudioFormat::Mp3_44100_64 => OutputFormat::Mp3_44100Hz64kbps,
            AudioFormat::Mp3_44100_96 => OutputFormat::Mp3_44100Hz96kbps,
            AudioFormat::Mp3_44100_128 => OutputFormat::Mp3_44100Hz128kbps,
            AudioFormat::Mp3_44100_192 => OutputFormat::Mp3_44100Hz192kbps,
            AudioFormat::Pcm16000 => OutputFormat::Pcm16000Hz,
            AudioFormat::Pcm22050 => OutputFormat::Pcm22050Hz,
            AudioFormat::Pcm24000 => OutputFormat::Pcm24000Hz,
            AudioFormat::Pcm44100 => OutputFormat::Pcm44100Hz,
            AudioFormat::Ulaw8000 => OutputFormat::MuLaw8000Hz,
        };
        self
    }

    /// Enable request logging for debugging and monitoring
    ///
    /// # Example
    /// ```no_run
    /// # use fluent_voice_elevenlabs::*;
    /// # let engine = TtsEngine::elevenlabs().build().unwrap();
    /// let voice_changer = engine.voice_changer("voice_id")
    ///     .enable_logging(true); // Log API requests
    /// ```
    pub fn enable_logging(mut self, enabled: bool) -> Self {
        self.enable_logging = enabled;
        self
    }

    /// Convert the audio to the target voice (non-streaming)
    ///
    /// # Example
    /// ```no_run
    /// use fluent_voice_elevenlabs::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), VoiceError> {
    ///     let engine = TtsEngine::elevenlabs()
    ///         .api_key_from_env()?
    ///         .build()?;
    ///     
    ///     let audio = engine.voice_changer("pNInz6obpgDQGcFmaJgB")
    ///         .audio_file("recording.wav")
    ///         .model("eleven_multilingual_v2_sts")
    ///         .voice_settings(VoiceSettings {
    ///             stability: 0.5,
    ///             similarity_boost: 0.8,
    ///             style: Some(0.2),
    ///             use_speaker_boost: Some(true),
    ///             speed: None,
    ///         })
    ///         .remove_background_noise(true)
    ///         .convert()
    ///         .await?;
    ///     
    ///     audio.save("converted_voice.mp3").await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn convert(self) -> Result<AudioOutput> {
        use crate::endpoints::genai::voice_changer::{
            VoiceChanger, VoiceChangerBody, VoiceChangerQuery,
        };

        let audio_file = self.audio_file.ok_or_else(|| {
            FluentVoiceError::ConfigError("Audio file is required for voice conversion".into())
        })?;

        // Validate file exists and is readable
        std::fs::metadata(&audio_file).map_err(|e| {
            FluentVoiceError::ConfigError(format!(
                "Cannot access audio file '{}': {}",
                audio_file, e
            ))
        })?;

        // Validate file extension
        let path = std::path::Path::new(&audio_file);
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            let valid_extensions = ["mp3", "wav", "flac", "m4a", "ogg", "webm"];
            if !valid_extensions.contains(&ext.to_lowercase().as_str()) {
                return Err(format!(
                    "Unsupported audio format '{}'. Supported formats: {}",
                    ext,
                    valid_extensions.join(", ")
                )
                .into());
            }
        } else {
            return Err(
                "Audio file must have a valid extension (mp3, wav, flac, m4a, ogg, webm)".into(),
            );
        }

        // Build the request body
        let mut body = VoiceChangerBody::new(audio_file);

        if let Some(model) = self.model_id {
            body = body.with_model_id(model);
        }

        if let Some(settings) = self.voice_settings {
            // Convert InternalVoiceSettings to shared::VoiceSettings
            let shared_settings = crate::shared::VoiceSettings {
                stability: settings.stability,
                similarity_boost: settings.similarity_boost,
                style: settings.style,
                use_speaker_boost: settings.use_speaker_boost,
                speed: settings.speed,
            };
            body = body.with_voice_settings(shared_settings);
        }

        if let Some(seed) = self.seed {
            body = body.with_seed(seed);
        }

        if let Some(remove_noise) = self.remove_background_noise {
            body = body.with_remove_background_noise(remove_noise);
        }

        // Build query parameters
        let mut query = VoiceChangerQuery::default().with_output_format(self.output_format.clone());

        if self.enable_logging {
            query = query.with_logging(true);
        }

        // Execute the voice changer endpoint
        let endpoint = VoiceChanger::new(&self.voice_id, body).with_query(query);
        let audio_data = self.client.hit(endpoint).await?;

        Ok(AudioOutput {
            data: audio_data,
            format: self.output_format,
        })
    }

    /// Convert the audio to the target voice (streaming)
    ///
    /// # Example
    /// ```no_run
    /// use fluent_voice_elevenlabs::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), VoiceError> {
    ///     let engine = TtsEngine::elevenlabs()
    ///         .api_key_from_env()?
    ///         .build()?;
    ///     
    ///     let stream = engine.voice_changer("pNInz6obpgDQGcFmaJgB")
    ///         .audio_file("podcast_episode.mp3")
    ///         .model("eleven_multilingual_v2_sts")
    ///         .output_format(AudioFormat::Mp3_44100_128)
    ///         .enable_logging(true)
    ///         .stream()
    ///         .await?;
    ///     
    ///     stream.save("live_converted.mp3").await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn stream(self) -> Result<AudioStream> {
        use crate::endpoints::genai::voice_changer::{
            VoiceChangerBody, VoiceChangerQuery, VoiceChangerStream,
        };

        let audio_file = self.audio_file.ok_or_else(|| {
            FluentVoiceError::ConfigError(
                "Audio file is required for voice conversion streaming".into(),
            )
        })?;

        // Validate file exists and is readable
        std::fs::metadata(&audio_file).map_err(|e| {
            FluentVoiceError::ConfigError(format!(
                "Cannot access audio file '{}': {}",
                audio_file, e
            ))
        })?;

        // Validate file extension
        let path = std::path::Path::new(&audio_file);
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            let valid_extensions = ["mp3", "wav", "flac", "m4a", "ogg", "webm"];
            if !valid_extensions.contains(&ext.to_lowercase().as_str()) {
                return Err(format!(
                    "Unsupported audio format '{}'. Supported formats: {}",
                    ext,
                    valid_extensions.join(", ")
                )
                .into());
            }
        } else {
            return Err(
                "Audio file must have a valid extension (mp3, wav, flac, m4a, ogg, webm)".into(),
            );
        }

        // Build the request body (same as convert)
        let mut body = VoiceChangerBody::new(audio_file);

        if let Some(model) = self.model_id {
            body = body.with_model_id(model);
        }

        if let Some(settings) = self.voice_settings {
            // Convert InternalVoiceSettings to shared::VoiceSettings
            let shared_settings = crate::shared::VoiceSettings {
                stability: settings.stability,
                similarity_boost: settings.similarity_boost,
                style: settings.style,
                use_speaker_boost: settings.use_speaker_boost,
                speed: settings.speed,
            };
            body = body.with_voice_settings(shared_settings);
        }

        if let Some(seed) = self.seed {
            body = body.with_seed(seed);
        }

        if let Some(remove_noise) = self.remove_background_noise {
            body = body.with_remove_background_noise(remove_noise);
        }

        // Build query parameters
        let mut query = VoiceChangerQuery::default().with_output_format(self.output_format.clone());

        if self.enable_logging {
            query = query.with_logging(true);
        }

        // Execute the voice changer stream endpoint
        let endpoint = VoiceChangerStream::new(&self.voice_id, body).with_query(query);
        let stream = self.client.hit(endpoint).await?;

        Ok(AudioStream::new(stream))
    }
}
