//! Core TtsConversationBuilderImpl struct and constructor methods

use super::conversation_impl::TtsConversationImpl;
use super::speaker_line::SpeakerLine;
use fluent_voice_domain::{
    audio_format::AudioFormat,
    language::Language,
    pronunciation_dict::{PronunciationDictId, RequestId},
    AudioChunk, VoiceError,
};
use futures::Stream;

/// Zero-allocation TTS conversation builder with arrow syntax support
pub struct TtsConversationBuilderImpl<AudioStream> {
    pub(super) lines: Vec<SpeakerLine>,
    pub(super) global_language: Option<Language>,
    pub(super) global_speed: Option<crate::vocal_speed::VocalSpeedMod>,
    pub(super) model: Option<crate::model_id::ModelId>,
    pub(super) stability: Option<crate::stability::Stability>,
    pub(super) similarity: Option<crate::similarity::Similarity>,
    pub(super) speaker_boost: Option<crate::speaker_boost::SpeakerBoost>,
    pub(super) style_exaggeration: Option<crate::style_exaggeration::StyleExaggeration>,
    pub(super) output_format: Option<AudioFormat>,
    pub(super) pronunciation_dictionaries: Vec<PronunciationDictId>,
    pub(super) seed: Option<u64>,
    pub(super) previous_text: Option<String>,
    pub(super) next_text: Option<String>,
    pub(super) previous_request_ids: Vec<RequestId>,
    pub(super) next_request_ids: Vec<RequestId>,
    pub(super) additional_params: std::collections::HashMap<String, String>,
    pub(super) metadata: std::collections::HashMap<String, String>,
    pub(super) voice_clone_path: Option<std::path::PathBuf>,
    pub(super) chunk_handler:
        Option<Box<dyn Fn(Result<AudioChunk, VoiceError>) -> AudioChunk + Send + Sync + 'static>>,
    pub(super) synth_fn:
        Option<Box<dyn FnOnce(&[SpeakerLine], Option<&Language>) -> AudioStream + Send>>,
    pub(super) prelude: Option<Box<dyn Fn() -> Vec<u8> + Send + 'static>>,
    pub(super) postlude: Option<Box<dyn Fn() -> Vec<u8> + Send + 'static>>,
    pub(super) engine_config: Option<hashbrown::HashMap<String, String>>,
    pub(super) result_handler: Option<
        Box<dyn FnOnce(Result<TtsConversationImpl<AudioStream>, VoiceError>) + Send + 'static>,
    >,
}

impl<AudioStream> TtsConversationBuilderImpl<AudioStream>
where
    AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin + 'static,
{
    /// Create a default TTS builder with working dia neural synthesis
    #[inline]
    pub fn default() -> TtsConversationBuilderImpl<crate::audio_stream::AudioStream>
    where
        AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin + 'static,
    {
        use candle_core::Device;
        use dia::voice::VoicePool;
        use std::sync::Arc;
        use tokio::sync::mpsc;
        use tokio_stream::wrappers::UnboundedReceiverStream;

        let synth_fn = |lines: &[SpeakerLine], _global_language: Option<&Language>| {
            let lines = lines.to_vec();
            let (tx, rx) = mpsc::unbounded_channel::<fluent_voice_domain::AudioChunk>();

            tokio::spawn(async move {
                let cache_dir = std::env::temp_dir().join("fluent_voice_cache");
                let pool = match VoicePool::new_with_config(cache_dir, Device::Cpu) {
                    Ok(pool) => Arc::new(pool),
                    Err(e) => {
                        let error_chunk = fluent_voice_domain::AudioChunk::with_metadata(
                            Vec::new(),
                            0,
                            0,
                            None,
                            Some(format!("[ERROR] Failed to create voice pool: {}", e)),
                            None,
                        );
                        let _ = tx.send(error_chunk);
                        return;
                    }
                };

                let mut cumulative_time_ms = 0u64;
                for line in lines.into_iter() {
                    match super::synthesis::synthesize_speech_internal(
                        &pool, &line.id, &line.text, None,
                    )
                    .await
                    {
                        Ok(audio_bytes) => {
                            let duration_ms = if audio_bytes.is_empty() {
                                0
                            } else {
                                let samples = audio_bytes.len() / 2;
                                let duration_secs = samples as f64 / 24000.0;
                                (duration_secs * 1000.0) as u64
                            };

                            let start_ms = cumulative_time_ms;
                            cumulative_time_ms += duration_ms;
                            let chunk = fluent_voice_domain::AudioChunk::with_metadata(
                                audio_bytes,
                                duration_ms,
                                start_ms,
                                Some(line.id.clone()),
                                Some(line.text.clone()),
                                Some(fluent_voice_domain::AudioFormat::Pcm24Khz),
                            );

                            if tx.send(chunk).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let error_chunk = fluent_voice_domain::AudioChunk::with_metadata(
                                Vec::new(),
                                0,
                                0,
                                None,
                                Some(format!("[ERROR] {}", e)),
                                None,
                            );
                            if tx.send(error_chunk).is_err() {
                                break;
                            }
                        }
                    }
                }
            });

            let stream = UnboundedReceiverStream::new(rx);
            crate::audio_stream::AudioStream::new(Box::pin(stream))
        };

        TtsConversationBuilderImpl::new(synth_fn)
    }

    #[inline]
    pub fn new<F>(synth_fn: F) -> Self
    where
        F: FnOnce(&[SpeakerLine], Option<&Language>) -> AudioStream + Send + 'static,
    {
        Self {
            lines: Vec::new(),
            global_language: None,
            global_speed: None,
            model: None,
            stability: None,
            similarity: None,
            speaker_boost: None,
            style_exaggeration: None,
            output_format: None,
            pronunciation_dictionaries: Vec::new(),
            seed: None,
            previous_text: None,
            next_text: None,
            previous_request_ids: Vec::new(),
            next_request_ids: Vec::new(),
            additional_params: std::collections::HashMap::new(),
            metadata: std::collections::HashMap::new(),
            voice_clone_path: None,
            chunk_handler: None,
            synth_fn: Some(Box::new(synth_fn)),
            prelude: None,
            postlude: None,
            engine_config: None,
            result_handler: None,
        }
    }

    #[inline]
    pub fn add_line(mut self, speaker: SpeakerLine) -> Self {
        self.lines.push(speaker);
        self
    }

    #[inline]
    pub fn with_prelude(mut self, prelude: impl Fn() -> Vec<u8> + Send + 'static) -> Self {
        self.prelude = Some(Box::new(prelude));
        self
    }

    #[inline]
    pub fn with_postlude(mut self, postlude: impl Fn() -> Vec<u8> + Send + 'static) -> Self {
        self.postlude = Some(Box::new(postlude));
        self
    }

    #[inline]
    pub fn with_speed(mut self, speed: crate::vocal_speed::VocalSpeedMod) -> Self {
        self.global_speed = Some(speed);
        self
    }

    #[inline]
    pub fn engine_config(
        mut self,
        config: impl Into<hashbrown::HashMap<&'static str, &'static str>>,
    ) -> Self {
        let config_map = config.into();
        let mut string_config = hashbrown::HashMap::new();
        for (k, v) in config_map {
            string_config.insert(k.to_string(), v.to_string());
        }
        self.engine_config = Some(string_config);
        self
    }

    #[inline]
    pub async fn finish_conversation(self) -> Result<TtsConversationImpl<AudioStream>, VoiceError> {
        match self.synth_fn {
            Some(synth_fn) => {
                let conversation = TtsConversationImpl {
                    lines: self.lines,
                    global_language: self.global_language,
                    global_speed: self.global_speed,
                    model: self.model,
                    stability: self.stability,
                    similarity: self.similarity,
                    speaker_boost: self.speaker_boost,
                    style_exaggeration: self.style_exaggeration,
                    output_format: self.output_format,
                    pronunciation_dictionaries: self.pronunciation_dictionaries,
                    seed: self.seed,
                    previous_text: self.previous_text,
                    next_text: self.next_text,
                    previous_request_ids: self.previous_request_ids,
                    next_request_ids: self.next_request_ids,
                    voice_clone_path: self.voice_clone_path,
                    synth_fn,
                };

                Ok(conversation)
            }
            None => Err(VoiceError::NotSynthesizable(
                "A synthesis function must be provided".to_string(),
            )),
        }
    }
}
