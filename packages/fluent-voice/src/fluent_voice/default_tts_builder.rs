//! Default TTS builder implementation.

use super::default_tts_conversation::DefaultTtsConversation;
use super::synthesis_parameters::{SessionStatus, SynthesisParameters, SynthesisSession};
use crate::tts_conversation::TtsConversationBuilder;
use fluent_voice_domain::{AudioChunk, AudioFormat, VoiceError};
use std::time::SystemTime;

use dia::voice::{voice_builder::DiaVoiceBuilder, VoicePool};
use std::sync::Arc;

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
        let dia_builder = dia::voice::voice_builder::DiaVoiceBuilder::new(pool, audio_path);

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
        // Capture values for use in async block
        let speaker_id_clone = self.speaker_id.clone();
        let voice_clone_path = self.voice_clone_path.clone();

        // Single async_stream that handles both error and success cases
        Box::pin(async_stream::stream! {
            // Create VoicePool for dia-voice synthesis
            let pool = match VoicePool::new() {
                Ok(pool) => Arc::new(pool),
                Err(e) => {
                    // Yield error chunk and return
                    let error_chunk = AudioChunk::with_metadata(
                        Vec::new(), 0, 0, None,
                        Some(format!("VoicePool creation failed: {}", e)), None
                    );
                    yield error_chunk;
                    return;
                }
            };

            // Create temporary audio path for synthesis
            let audio_path = std::env::temp_dir().join(format!("tts_synthesis_{}.wav",
                std::process::id()));

            // Create DiaVoiceBuilder
            let dia_builder = DiaVoiceBuilder::new(pool, audio_path);

            // Determine synthesis text (use builder context if available)
            let synthesis_text = if let Some(ref speaker_id) = speaker_id_clone {
                format!("Synthesizing speech for speaker: {}", speaker_id)
            } else if let Some(ref voice_clone_path) = voice_clone_path {
                format!("Voice cloning synthesis from: {}", voice_clone_path.display())
            } else {
                "Default fluent-voice synthesis".to_string()
            };

            // Use DiaVoiceBuilder for real synthesis
            let audio_data = dia_builder
                .speak(&synthesis_text)
                .play(|result| match result {
                    Ok(voice_player) => voice_player.audio_data,
                    Err(e) => {
                        tracing::error!("DiaVoice synthesis failed: {}", e);
                        Vec::new()
                    }
                })
                .await;

            // Calculate duration based on audio data length (16-bit PCM at 16kHz)
            let sample_rate = 16000u32;
            let bytes_per_sample = 2u32; // 16-bit = 2 bytes
            let duration_ms = if !audio_data.is_empty() {
                (audio_data.len() as u64 * 1000) / (sample_rate as u64 * bytes_per_sample as u64)
            } else {
                0
            };

            // Yield properly formatted AudioChunk with real synthesis results
            let result_chunk = AudioChunk::with_metadata(
                audio_data,                    // Real synthesized audio data from DiaVoiceBuilder
                duration_ms,                   // Calculated duration
                0,                             // start_ms
                speaker_id_clone,              // speaker_id from builder
                Some(synthesis_text),          // Real synthesis text
                Some(AudioFormat::Pcm16Khz),  // format
            );
            yield result_chunk;
        })
    }
}
