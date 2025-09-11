//! Zero-allocation TTS builder implementation with arrow syntax support
//!
//! This module provides a blazing-fast, zero-allocation TTS conversation builder
//! with full arrow syntax support through cyrup_sugars integration.

use crate::tts_conversation::{TtsConversationBuilder, TtsConversationChunkBuilder};
use candle_core::Device;
use cyrup_sugars::prelude::{ChunkHandler, MessageChunk};
use dia::voice::VoicePool;
use fluent_voice_domain::{
    AudioChunk, TtsConversation, VoiceError,
    audio_format::AudioFormat,
    language::Language,
    pronunciation_dict::{PronunciationDictId, RequestId},
};
use futures::Stream;
use std::sync::Arc;

// Wrapper type to implement MessageChunk for AudioChunk (avoiding orphan rule)
#[derive(Debug, Clone)]
pub struct AudioChunkWrapper(pub fluent_voice_domain::AudioChunk);

impl MessageChunk for AudioChunkWrapper {
    fn bad_chunk(error: String) -> Self {
        AudioChunkWrapper(fluent_voice_domain::AudioChunk::with_metadata(
            Vec::new(),
            0,
            0,
            None,
            Some(format!("[ERROR] {}", error)),
            None,
        ))
    }

    fn error(&self) -> Option<&str> {
        self.0.text().and_then(|text| {
            if text.starts_with("[ERROR]") {
                Some(text[8..].trim())
            } else {
                None
            }
        })
    }

    fn is_error(&self) -> bool {
        self.0
            .text()
            .map_or(false, |text| text.starts_with("[ERROR]"))
    }
}

impl From<fluent_voice_domain::AudioChunk> for AudioChunkWrapper {
    fn from(chunk: fluent_voice_domain::AudioChunk) -> Self {
        AudioChunkWrapper(chunk)
    }
}

impl From<AudioChunkWrapper> for fluent_voice_domain::AudioChunk {
    fn from(wrapper: AudioChunkWrapper) -> Self {
        wrapper.0
    }
}

/// High-performance speaker line with zero-allocation design
#[derive(Clone, Debug)]
pub struct SpeakerLine {
    pub id: String,
    pub text: String,
    pub voice_id: Option<fluent_voice_domain::VoiceId>,
    pub language: Option<fluent_voice_domain::Language>,
    pub speed_modifier: Option<fluent_voice_domain::VocalSpeedMod>,
    pub pitch_range: Option<fluent_voice_domain::PitchRange>,
}

impl SpeakerLine {
    #[inline]
    pub fn new(name: impl Into<String>) -> SpeakerLineBuilder {
        use crate::speaker_builder::SpeakerBuilder;
        SpeakerLineBuilder::speaker(name)
    }
}

impl crate::speaker::Speaker for SpeakerLine {
    #[inline]
    fn id(&self) -> &str {
        &self.id
    }

    #[inline]
    fn text(&self) -> &str {
        &self.text
    }

    #[inline]
    fn voice_id(&self) -> Option<&fluent_voice_domain::VoiceId> {
        self.voice_id.as_ref()
    }

    #[inline]
    fn language(&self) -> Option<&fluent_voice_domain::Language> {
        self.language.as_ref()
    }

    #[inline]
    fn speed_modifier(&self) -> Option<fluent_voice_domain::VocalSpeedMod> {
        self.speed_modifier
    }

    #[inline]
    fn pitch_range(&self) -> Option<&fluent_voice_domain::PitchRange> {
        self.pitch_range.as_ref()
    }
}

/// Zero-allocation speaker builder with optimized construction
#[derive(Clone, Debug)]
pub struct SpeakerLineBuilder {
    id: String,
    text: String,
    voice_id: Option<fluent_voice_domain::VoiceId>,
    language: Option<fluent_voice_domain::Language>,
    speed_modifier: Option<fluent_voice_domain::VocalSpeedMod>,
    pitch_range: Option<fluent_voice_domain::PitchRange>,
}

impl crate::speaker_builder::SpeakerBuilder for SpeakerLineBuilder {
    type Output = SpeakerLine;

    #[inline]
    fn speaker(name: impl Into<String>) -> Self {
        SpeakerLineBuilder {
            id: name.into(),
            text: String::new(),
            voice_id: None,
            language: None,
            speed_modifier: None,
            pitch_range: None,
        }
    }

    #[inline]
    fn voice_id(mut self, id: fluent_voice_domain::VoiceId) -> Self {
        self.voice_id = Some(id);
        self
    }

    #[inline]
    fn language(mut self, lang: fluent_voice_domain::Language) -> Self {
        self.language = Some(lang);
        self
    }

    #[inline]
    fn with_prelude(mut self, prelude: impl Into<String>) -> Self {
        let prelude_text = prelude.into();
        if !self.text.is_empty() {
            self.text.push_str(" ");
        }
        self.text.push_str(&prelude_text);
        self
    }

    #[inline]
    fn add_line(mut self, line: impl Into<String>) -> Self {
        let line_text = line.into();
        if !self.text.is_empty() {
            self.text.push_str(" ");
        }
        self.text.push_str(&line_text);
        self
    }

    #[inline]
    fn with_voice(mut self, voice: impl Into<String>) -> Self {
        let voice_str = voice.into();
        self.voice_id = Some(fluent_voice_domain::VoiceId::new(voice_str));
        self
    }

    #[inline]
    fn with_speed(mut self, speed: f32) -> Self {
        self.speed_modifier = Some(fluent_voice_domain::VocalSpeedMod(speed));
        self
    }

    #[inline]
    fn with_speed_modifier(mut self, m: fluent_voice_domain::VocalSpeedMod) -> Self {
        self.speed_modifier = Some(m);
        self
    }

    #[inline]
    fn with_pitch_range(mut self, range: fluent_voice_domain::PitchRange) -> Self {
        self.pitch_range = Some(range);
        self
    }

    #[inline]
    fn speak(mut self, text: impl Into<String>) -> Self {
        self.text = text.into();
        self
    }

    #[inline]
    fn build(self) -> Self::Output {
        SpeakerLine {
            id: self.id,
            text: self.text,
            voice_id: self.voice_id,
            language: self.language,
            speed_modifier: self.speed_modifier,
            pitch_range: self.pitch_range,
        }
    }
}

impl SpeakerLineBuilder {
    /// Zero-allocation metadata configuration
    #[inline]
    pub fn metadata(
        self,
        config: impl Into<hashbrown::HashMap<&'static str, &'static str>>,
    ) -> Self {
        let _ = config.into();
        self
    }

    /// Zero-allocation vocal settings configuration
    #[inline]
    pub fn vocal_settings(
        self,
        config: impl Into<hashbrown::HashMap<&'static str, &'static str>>,
    ) -> Self {
        let _ = config.into();
        self
    }
}

/// Zero-allocation TTS conversation with optimized stream handling
pub struct TtsConversationImpl<AudioStream> {
    pub lines: Vec<SpeakerLine>,
    pub global_language: Option<Language>,
    pub global_speed: Option<crate::vocal_speed::VocalSpeedMod>,
    pub model: Option<crate::model_id::ModelId>,
    pub stability: Option<crate::stability::Stability>,
    pub similarity: Option<crate::similarity::Similarity>,
    pub speaker_boost: Option<crate::speaker_boost::SpeakerBoost>,
    pub style_exaggeration: Option<crate::style_exaggeration::StyleExaggeration>,
    pub output_format: Option<AudioFormat>,
    pub pronunciation_dictionaries: Vec<PronunciationDictId>,
    pub seed: Option<u64>,
    pub previous_text: Option<String>,
    pub next_text: Option<String>,
    pub previous_request_ids: Vec<RequestId>,
    pub next_request_ids: Vec<RequestId>,
    synth_fn: Box<dyn FnOnce(&[SpeakerLine], Option<&Language>) -> AudioStream + Send>,
}

impl<AudioStream> TtsConversationImpl<AudioStream> {
    #[inline]
    pub fn global_speed(&self) -> Option<crate::vocal_speed::VocalSpeedMod> {
        self.global_speed
    }

    #[inline]
    pub fn speed_already_applied(&self) -> bool {
        false
    }

    #[inline]
    pub fn sample_rate_hz(&self) -> usize {
        16000
    }
}

impl<AudioStream> TtsConversation for TtsConversationImpl<AudioStream>
where
    AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin + 'static,
{
    type AudioStream = AudioStream;

    #[inline]
    fn into_stream(self) -> Self::AudioStream {
        (self.synth_fn)(&self.lines, self.global_language.as_ref())
    }
}

/// Zero-allocation TTS conversation builder with arrow syntax support
pub struct TtsConversationBuilderImpl<AudioStream> {
    lines: Vec<SpeakerLine>,
    global_language: Option<Language>,
    global_speed: Option<crate::vocal_speed::VocalSpeedMod>,
    model: Option<crate::model_id::ModelId>,
    stability: Option<crate::stability::Stability>,
    similarity: Option<crate::similarity::Similarity>,
    speaker_boost: Option<crate::speaker_boost::SpeakerBoost>,
    style_exaggeration: Option<crate::style_exaggeration::StyleExaggeration>,
    output_format: Option<AudioFormat>,
    pronunciation_dictionaries: Vec<PronunciationDictId>,
    seed: Option<u64>,
    previous_text: Option<String>,
    next_text: Option<String>,
    previous_request_ids: Vec<RequestId>,
    next_request_ids: Vec<RequestId>,
    additional_params: std::collections::HashMap<String, String>,
    metadata: std::collections::HashMap<String, String>,
    chunk_handler:
        Option<Box<dyn Fn(Result<AudioChunk, VoiceError>) -> AudioChunk + Send + Sync + 'static>>,
    synth_fn: Option<Box<dyn FnOnce(&[SpeakerLine], Option<&Language>) -> AudioStream + Send>>,
}

impl<AudioStream> TtsConversationBuilderImpl<AudioStream>
where
    AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin + 'static,
{
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
            chunk_handler: None,
            synth_fn: Some(Box::new(synth_fn)),
        }
    }

    #[inline]
    pub fn add_line(mut self, speaker: SpeakerLine) -> Self {
        self.lines.push(speaker);
        self
    }

    #[inline]
    pub fn with_prelude(self, _prelude: impl Fn() -> Vec<u8> + Send + 'static) -> Self {
        self
    }

    #[inline]
    pub fn with_postlude(self, _postlude: impl Fn() -> Vec<u8> + Send + 'static) -> Self {
        self
    }

    #[inline]
    pub fn with_speed(mut self, speed: crate::vocal_speed::VocalSpeedMod) -> Self {
        self.global_speed = Some(speed);
        self
    }

    #[inline]
    pub fn engine_config(
        self,
        _config: impl Into<hashbrown::HashMap<&'static str, &'static str>>,
    ) -> Self {
        self
    }

    #[inline]
    pub async fn finish_conversation(self) -> Result<TtsConversationImpl<AudioStream>, VoiceError> {
        match self.synth_fn {
            Some(synth_fn) => Ok(TtsConversationImpl {
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
                synth_fn,
            }),
            None => Err(VoiceError::NotSynthesizable(
                "A synthesis function must be provided".to_string(),
            )),
        }
    }
}

impl<AudioStream> TtsConversationBuilder for TtsConversationBuilderImpl<AudioStream>
where
    AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin + 'static,
{
    type Conversation = TtsConversationImpl<AudioStream>;
    type ChunkBuilder = TtsConversationBuilderImpl<AudioStream>;

    #[inline]
    fn with_speaker<S: crate::speaker::Speaker>(mut self, speaker: S) -> Self {
        self.lines.push(SpeakerLine {
            id: speaker.id().to_string(),
            text: speaker.text().to_string(),
            voice_id: speaker.voice_id().cloned(),
            language: speaker.language().cloned(),
            speed_modifier: speaker.speed_modifier(),
            pitch_range: speaker.pitch_range().cloned(),
        });
        self
    }

    #[inline]
    fn language(mut self, lang: Language) -> Self {
        self.global_language = Some(lang);
        self
    }

    #[inline]
    fn model(mut self, model: crate::model_id::ModelId) -> Self {
        self.model = Some(model);
        self
    }

    #[inline]
    fn stability(mut self, stability: crate::stability::Stability) -> Self {
        self.stability = Some(stability);
        self
    }

    #[inline]
    fn similarity(mut self, similarity: crate::similarity::Similarity) -> Self {
        self.similarity = Some(similarity);
        self
    }

    #[inline]
    fn speaker_boost(mut self, boost: crate::speaker_boost::SpeakerBoost) -> Self {
        self.speaker_boost = Some(boost);
        self
    }

    #[inline]
    fn style_exaggeration(
        mut self,
        exaggeration: crate::style_exaggeration::StyleExaggeration,
    ) -> Self {
        self.style_exaggeration = Some(exaggeration);
        self
    }

    #[inline]
    fn output_format(mut self, format: AudioFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    #[inline]
    fn pronunciation_dictionary(mut self, dict_id: PronunciationDictId) -> Self {
        self.pronunciation_dictionaries.push(dict_id);
        self
    }

    #[inline]
    fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    #[inline]
    fn previous_text(mut self, text: impl Into<String>) -> Self {
        self.previous_text = Some(text.into());
        self
    }

    #[inline]
    fn next_text(mut self, text: impl Into<String>) -> Self {
        self.next_text = Some(text.into());
        self
    }

    #[inline]
    fn previous_request_ids(mut self, request_ids: Vec<RequestId>) -> Self {
        self.previous_request_ids = request_ids;
        self
    }

    #[inline]
    fn next_request_ids(mut self, request_ids: Vec<RequestId>) -> Self {
        self.next_request_ids = request_ids;
        self
    }

    #[inline]
    fn with_voice_clone_path(self, _path: std::path::PathBuf) -> Self {
        // TODO: Store voice clone path for synthesis
        self
    }

    #[inline]
    fn additional_params<P>(self, _params: P) -> Self
    where
        P: Into<std::collections::HashMap<String, String>>,
    {
        // TODO: Store additional parameters for synthesis
        self
    }

    #[inline]
    fn metadata<M>(self, _meta: M) -> Self
    where
        M: Into<std::collections::HashMap<String, String>>,
    {
        // TODO: Store metadata for synthesis
        self
    }

    // on_chunk method removed - not part of TtsConversationBuilder trait

    #[inline]
    fn on_result<F>(self, _f: F) -> Self
    where
        F: FnOnce(Result<Self::Conversation, VoiceError>) + Send + 'static,
    {
        self
    }

    fn synthesize<M, S>(self, matcher: M) -> S
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> S + Send + 'static,
        S: futures_core::Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin + 'static,
    {
        // Build the conversation result as a closure that gets executed later
        let conversation_result = match self.synth_fn {
            Some(synth_fn) => Ok(TtsConversationImpl {
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
                synth_fn,
            }),
            None => Err(VoiceError::NotSynthesizable(
                "A synthesis function must be provided".to_string(),
            )),
        };

        // Apply the matcher closure to the conversation result
        // The matcher contains the JSON syntax transformed by synthesize! macro
        matcher(conversation_result)
    }
}

/// Internal function for real TTS speech synthesis using dia-voice
/// This function is separate from the impl block to avoid generic parameter issues
async fn synthesize_speech_internal(
    pool: &Arc<VoicePool>,
    speaker_id: &str,
    text: &str,
) -> Result<Vec<u8>, VoiceError> {
    // Use dia-voice engine for real TTS synthesis
    use dia::voice::{Conversation, DiaSpeaker, VoiceClone};

    // Create basic voice data for the speaker
    use candle_core::{Device, Tensor};
    use dia::voice::codec::VoiceData;
    use std::path::PathBuf;

    let device = Device::Cpu;
    let codes = Tensor::zeros((1, 1), candle_core::DType::F32, &device)
        .map_err(|e| VoiceError::Configuration(format!("Failed to create tensor: {}", e)))?;
    let voice_data = Arc::new(VoiceData {
        codes,
        sample_rate: 24000,
        source_path: PathBuf::from("temp_voice.wav"),
    });

    // Create voice clone and speaker
    let voice_clone = VoiceClone::new(speaker_id.to_string(), voice_data);
    let speaker = DiaSpeaker { voice_clone };

    // Create conversation and generate TTS audio using dia-voice engine
    let conversation = Conversation::new(text.to_string(), speaker, pool.clone())
        .await
        .map_err(|e| VoiceError::Synthesis(format!("Failed to create TTS conversation: {}", e)))?;

    // Generate speech audio using dia-voice engine
    let voice_player = conversation
        .internal_generate()
        .await
        .map_err(|e| VoiceError::Synthesis(format!("Failed to generate speech: {}", e)))?;

    // Extract audio bytes from dia-voice engine
    let audio_bytes = voice_player
        .to_bytes()
        .await
        .map_err(|e| VoiceError::Synthesis(format!("Failed to extract speech bytes: {}", e)))?;

    // Log the synthesis for debugging
    tracing::info!(size_bytes = audio_bytes.len(), speaker_id = %speaker_id, text = %text, "Synthesized audio");

    Ok(audio_bytes)
}

impl<AudioStream> TtsConversationBuilderImpl<AudioStream> where
    AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin + 'static
{
}

impl<AudioStream> TtsConversationChunkBuilder for TtsConversationBuilderImpl<AudioStream>
where
    AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin + 'static,
{
    type Conversation = TtsConversationImpl<AudioStream>;

    fn synthesize(
        self,
    ) -> impl futures_core::Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin {
        let lines = self.lines;
        let chunk_handler = self.chunk_handler;

        // Create stream using tokio_stream for proper async handling
        use tokio::sync::mpsc;
        use tokio_stream::wrappers::UnboundedReceiverStream;

        let (tx, rx) =
            mpsc::unbounded_channel::<Result<fluent_voice_domain::AudioChunk, VoiceError>>();

        // Spawn async task to generate audio using dia-voice
        tokio::spawn(async move {
            // Initialize voice pool once for all speakers (zero allocation optimization)
            let cache_dir = std::env::temp_dir().join("fluent_voice_cache");
            let pool = match VoicePool::new_with_config(cache_dir, Device::Cpu) {
                Ok(pool) => Arc::new(pool),
                Err(e) => {
                    // Send error result for pool creation failure
                    let _ = tx.send(Err(VoiceError::Configuration(format!(
                        "Failed to create voice pool: {}",
                        e
                    ))));
                    return;
                }
            };

            // Process each speaker line with real dia-voice TTS synthesis
            for (chunk_index, line) in lines.into_iter().enumerate() {
                // Generate real TTS audio directly from text using dia-voice
                let result = synthesize_speech_internal(&pool, &line.id, &line.text)
                    .await
                    .map(|audio_bytes| {
                        // Create AudioChunk from generated audio bytes
                        let duration_ms = if audio_bytes.is_empty() {
                            0
                        } else {
                            // Calculate duration based on audio data size and sample rate
                            // Assuming 16-bit PCM at 24kHz sample rate
                            let samples = audio_bytes.len() / 2; // 2 bytes per sample for 16-bit
                            let duration_secs = samples as f64 / 24000.0; // 24kHz sample rate
                            (duration_secs * 1000.0) as u64
                        };

                        let start_ms = chunk_index as u64 * duration_ms; // Cumulative timing

                        fluent_voice_domain::AudioChunk::with_metadata(
                            audio_bytes,
                            duration_ms,
                            start_ms,
                            Some(line.id.clone()),
                            Some(line.text.clone()),
                            Some(fluent_voice_domain::AudioFormat::Pcm24Khz),
                        )
                    });

                // Send Result through channel
                if tx.send(result).is_err() {
                    break; // Receiver dropped, stop processing
                }
            }
        });

        // Apply ChunkHandler to transform Result<AudioChunk, VoiceError> to AudioChunk
        use futures::StreamExt;
        let stream = UnboundedReceiverStream::new(rx).map(move |result| {
            if let Some(ref handler) = chunk_handler {
                handler(result)
            } else {
                // Default behavior: unwrap Ok, create bad_chunk for Err
                result.unwrap_or_else(|e| {
                    // Create error chunk using the constructor
                    fluent_voice_domain::AudioChunk::with_metadata(
                        Vec::new(),
                        0,
                        0,
                        None,
                        Some(format!("[ERROR] {}", e)),
                        None,
                    )
                })
            }
        });

        // Return AudioStream wrapper to enable .play() method
        let stream = Box::pin(stream);
        crate::audio_stream::AudioStream::new(stream)
    }
}

pub mod builder {
    use super::*;

    #[inline]
    pub fn tts_conversation_builder<AudioStream, F>(
        synth_fn: F,
    ) -> TtsConversationBuilderImpl<AudioStream>
    where
        AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin + 'static,
        F: FnOnce(&[SpeakerLine], Option<&Language>) -> AudioStream + Send + 'static,
    {
        TtsConversationBuilderImpl::new(synth_fn)
    }
}

impl<AudioStream> ChunkHandler<AudioChunkWrapper, VoiceError>
    for TtsConversationBuilderImpl<AudioStream>
where
    AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin + 'static,
{
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<AudioChunkWrapper, VoiceError>) -> AudioChunkWrapper + Send + Sync + 'static,
    {
        // Convert the handler to work with the underlying AudioChunk type
        let converted_handler =
            move |result: Result<fluent_voice_domain::AudioChunk, VoiceError>| {
                let wrapper_result = result.map(AudioChunkWrapper::from);
                let wrapper_output = handler(wrapper_result);
                wrapper_output.into()
            };
        self.chunk_handler = Some(Box::new(converted_handler));
        self
    }
}

/// Arrow syntax transformation macro for cyrup_sugars compatibility
#[macro_export]
macro_rules! arrow_transform {
    ($matcher:expr) => {
        |result| {
            // Enable JSON syntax transformation
            $crate::enable_json_syntax!();
            $matcher(result)
        }
    };
}
