//! Concrete TTS builder implementation
//!
//! This module provides a non-macro implementation of the TTS conversation builder
//! that can be used as a base for engine-specific implementations.

use crate::tts_conversation::{TtsConversation, TtsConversationBuilder};
use crate::audio_chunk::i16_stream_to_bytes_stream;
use fluent_voice_domain::{
    VoiceError,
    audio_format::AudioFormat,
    language::Language,
    pronunciation_dict::{PronunciationDictId, RequestId},
};
use futures_core::Stream;


/// Concrete speaker implementation for TTS operations.
#[derive(Clone, Debug)]
pub struct SpeakerLine {
    /// Speaker identifier
    pub id: String,
    /// Text to be spoken
    pub text: String,
    /// Optional voice ID for this speaker
    pub voice_id: Option<fluent_voice_domain::VoiceId>,
    /// Optional language override for this speaker
    pub language: Option<fluent_voice_domain::Language>,
    /// Optional speaking-rate multiplier
    pub speed_modifier: Option<fluent_voice_domain::VocalSpeedMod>,
    /// Optional pitch range for this speaker
    pub pitch_range: Option<fluent_voice_domain::PitchRange>,
}

impl SpeakerLine {
    /// Start building a new speaker with the given name.
    pub fn speaker(name: impl Into<String>) -> SpeakerLineBuilder {
        <SpeakerLineBuilder as crate::speaker_builder::SpeakerBuilder>::speaker(name)
    }
}

impl crate::speaker::Speaker for SpeakerLine {
    fn id(&self) -> &str {
        &self.id
    }

    fn text(&self) -> &str {
        &self.text
    }

    fn voice_id(&self) -> Option<&fluent_voice_domain::VoiceId> {
        self.voice_id.as_ref()
    }

    fn language(&self) -> Option<&fluent_voice_domain::Language> {
        self.language.as_ref()
    }

    fn speed_modifier(&self) -> Option<fluent_voice_domain::VocalSpeedMod> {
        self.speed_modifier
    }

    fn pitch_range(&self) -> Option<&fluent_voice_domain::PitchRange> {
        self.pitch_range.as_ref()
    }
}

/// Concrete builder for speaker configuration.
#[derive(Clone, Debug)]
pub struct SpeakerLineBuilder {
    id: String,
    text: String,
    voice_id: Option<fluent_voice_domain::VoiceId>,
    language: Option<fluent_voice_domain::Language>,
    speed_modifier: Option<fluent_voice_domain::VocalSpeedMod>,
    pitch_range: Option<fluent_voice_domain::PitchRange>,
    metadata: std::collections::HashMap<String, String>,
    vocal_settings: std::collections::HashMap<String, String>,
}

impl crate::speaker_builder::SpeakerBuilder for SpeakerLineBuilder {
    type Output = SpeakerLine;

    fn speaker(name: impl Into<String>) -> Self {
        SpeakerLineBuilder {
            id: name.into(),
            text: String::new(),
            voice_id: None,
            language: None,
            speed_modifier: None,
            pitch_range: None,
            metadata: std::collections::HashMap::new(),
            vocal_settings: std::collections::HashMap::new(),
        }
    }

    fn voice_id(mut self, id: fluent_voice_domain::VoiceId) -> Self {
        self.voice_id = Some(id);
        self
    }

    fn language(mut self, lang: fluent_voice_domain::Language) -> Self {
        self.language = Some(lang);
        self
    }

    fn with_speed_modifier(mut self, m: fluent_voice_domain::VocalSpeedMod) -> Self {
        self.speed_modifier = Some(m);
        self
    }

    fn with_pitch_range(mut self, range: fluent_voice_domain::PitchRange) -> Self {
        self.pitch_range = Some(range);
        self
    }

    fn speak(mut self, text: impl Into<String>) -> Self {
        self.text = text.into();
        self
    }

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
    /// Configure speaker metadata using JSON object syntax
    pub fn metadata(mut self, config: impl Into<hashbrown::HashMap<&'static str, &'static str>>) -> Self {
        let meta_map = config.into();
        for (k, v) in meta_map {
            self.metadata.insert(k.to_string(), v.to_string());
        }
        self
    }

    /// Configure vocal settings using JSON object syntax
    pub fn vocal_settings(mut self, config: impl Into<hashbrown::HashMap<&'static str, &'static str>>) -> Self {
        let settings_map = config.into();
        for (k, v) in settings_map {
            self.vocal_settings.insert(k.to_string(), v.to_string());
        }
        self
    }
}

/// Implementation of SpeakerExt for all Speaker types using fluent-voice concrete builders


/// Concrete TTS conversation implementation.
pub struct TtsConversationImpl<AudioStream> {
    /// Lines to be spoken in the conversation
    pub lines: Vec<SpeakerLine>,
    /// Global language setting for the conversation
    pub global_language: Option<Language>,
    /// Global speaking rate setting
    pub global_speed: Option<crate::vocal_speed::VocalSpeedMod>,
    /// Model ID to use for synthesis
    pub model: Option<crate::model_id::ModelId>,
    /// Voice stability setting (0.0-1.0)
    pub stability: Option<crate::stability::Stability>,
    /// Voice similarity setting (0.0-1.0)
    pub similarity: Option<crate::similarity::Similarity>,
    /// Speaker boost setting (enhance speaker separation)
    pub speaker_boost: Option<crate::speaker_boost::SpeakerBoost>,
    /// Style exaggeration setting (0.0-1.0)
    pub style_exaggeration: Option<crate::style_exaggeration::StyleExaggeration>,
    /// Output audio format
    pub output_format: Option<AudioFormat>,
    /// Pronunciation dictionaries
    pub pronunciation_dictionaries: Vec<PronunciationDictId>,
    /// Deterministic seed
    pub seed: Option<u64>,
    /// Previous text for context
    pub previous_text: Option<String>,
    /// Next text for context
    pub next_text: Option<String>,
    /// Previous request IDs
    pub previous_request_ids: Vec<RequestId>,
    /// Next request IDs
    pub next_request_ids: Vec<RequestId>,
    /// Function to convert conversation to audio stream
    synth_fn: Box<dyn FnOnce(&[SpeakerLine], Option<&Language>) -> AudioStream + Send>,
}

impl<AudioStream> TtsConversationImpl<AudioStream> {
    /// Get the desired speed for the conversation
    pub fn global_speed(&self) -> Option<crate::vocal_speed::VocalSpeedMod> {
        self.global_speed
    }

    /// Check if speed has already been applied by the engine
    pub fn speed_already_applied(&self) -> bool {
        false // Engines override this
    }

    /// Get the sample rate of the audio stream
    pub fn sample_rate_hz(&self) -> usize {
        16000 // Engines override this
    }
}

impl<AudioStream> TtsConversation for TtsConversationImpl<AudioStream>
where
    AudioStream: Stream<Item = i16> + Send + Unpin + 'static,
{
    type AudioStream = AudioStream;

    fn into_stream(self) -> Self::AudioStream {
        (self.synth_fn)(&self.lines, self.global_language.as_ref())
    }
}

/// Concrete TTS conversation builder implementation.
pub struct TtsConversationBuilderImpl<AudioStream> {
    /// Lines to be spoken in the conversation
    lines: Vec<SpeakerLine>,
    /// Global language setting for the conversation
    global_language: Option<Language>,
    /// Global speaking rate setting
    global_speed: Option<crate::vocal_speed::VocalSpeedMod>,
    /// Engine configuration parameters
    engine_config: std::collections::HashMap<String, String>,
    /// Model ID to use for synthesis
    model: Option<crate::model_id::ModelId>,
    /// Voice stability setting (0.0-1.0)
    stability: Option<crate::stability::Stability>,
    /// Voice similarity setting (0.0-1.0)
    similarity: Option<crate::similarity::Similarity>,
    /// Speaker boost setting (enhance speaker separation)
    speaker_boost: Option<crate::speaker_boost::SpeakerBoost>,
    /// Style exaggeration setting (0.0-1.0)
    style_exaggeration: Option<crate::style_exaggeration::StyleExaggeration>,
    /// Output audio format
    output_format: Option<AudioFormat>,
    /// Pronunciation dictionaries
    pronunciation_dictionaries: Vec<PronunciationDictId>,
    /// Deterministic seed
    seed: Option<u64>,
    /// Previous text for context
    previous_text: Option<String>,
    /// Next text for context
    next_text: Option<String>,
    /// Previous request IDs
    previous_request_ids: Vec<RequestId>,
    /// Next request IDs
    next_request_ids: Vec<RequestId>,
    /// Function to synthesize audio from the conversation
    synth_fn: Box<dyn FnOnce(&[SpeakerLine], Option<&Language>) -> AudioStream + Send>,
    /// Chunk processor function for error handling
    chunk_processor: Option<Box<dyn FnMut(Result<Vec<u8>, VoiceError>) -> Vec<u8> + Send>>,
    /// Result processor function for custom error handling
    result_processor: Option<Box<dyn FnMut(VoiceError) -> Vec<u8> + Send>>,
}

impl<AudioStream> TtsConversationBuilderImpl<AudioStream>
where
    AudioStream: Stream<Item = i16> + Send + Unpin + 'static,
{
    /// Create a new TTS conversation builder with a custom synthesis function.
    pub fn new<F>(synth_fn: F) -> Self
    where
        F: FnOnce(&[SpeakerLine], Option<&Language>) -> AudioStream + Send + 'static,
    {
        Self {
            lines: Vec::new(),
            global_language: None,
            global_speed: None,
            engine_config: std::collections::HashMap::new(),
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
            synth_fn: Box::new(synth_fn),
            chunk_processor: None,
            result_processor: None,
        }
    }

    /// Set the global speaking rate
    pub fn with_speed(mut self, speed: crate::vocal_speed::VocalSpeedMod) -> Self {
        self.global_speed = Some(speed);
        self
    }

    /// Set the model ID for synthesis
    pub fn model(mut self, model: crate::model_id::ModelId) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the voice stability (0.0-1.0)
    pub fn stability(mut self, stability: crate::stability::Stability) -> Self {
        self.stability = Some(stability);
        self
    }

    /// Set the voice similarity to original (0.0-1.0)
    pub fn similarity(mut self, similarity: crate::similarity::Similarity) -> Self {
        self.similarity = Some(similarity);
        self
    }

    /// Enable or disable speaker boost
    pub fn speaker_boost(mut self, boost: crate::speaker_boost::SpeakerBoost) -> Self {
        self.speaker_boost = Some(boost);
        self
    }

    /// Set the style exaggeration level (0.0-1.0)
    pub fn style_exaggeration(
        mut self,
        exaggeration: crate::style_exaggeration::StyleExaggeration,
    ) -> Self {
        self.style_exaggeration = Some(exaggeration);
        self
    }

    /// Set the output audio format
    pub fn output_format(mut self, format: AudioFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    /// Configure engine parameters using JSON object syntax
    pub fn engine_config(mut self, config: impl Into<hashbrown::HashMap<&'static str, &'static str>>) -> Self {
        let config_map = config.into();
        for (k, v) in config_map {
            self.engine_config.insert(k.to_string(), v.to_string());
        }
        self
    }

    /// Add a pronunciation dictionary
    pub fn pronunciation_dictionary(mut self, dict_id: PronunciationDictId) -> Self {
        self.pronunciation_dictionaries.push(dict_id);
        self
    }

    /// Set deterministic seed
    pub fn seed(mut self, seed: u64) -> Self {
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

    /// Set previous request IDs
    pub fn previous_request_ids(mut self, request_ids: Vec<RequestId>) -> Self {
        self.previous_request_ids = request_ids;
        self
    }

    /// Set next request IDs
    pub fn next_request_ids(mut self, request_ids: Vec<RequestId>) -> Self {
        self.next_request_ids = request_ids;
        self
    }

    /// Helper for the macro-less implementation - completes the conversation
    pub async fn finish_conversation(self) -> Result<TtsConversationImpl<AudioStream>, VoiceError> {
        Ok(TtsConversationImpl {
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
            synth_fn: self.synth_fn,
        })
    }
}

impl<AudioStream> TtsConversationBuilder for TtsConversationBuilderImpl<AudioStream>
where
    AudioStream: Stream<Item = i16> + Send + Unpin + 'static,
{
    type Conversation = TtsConversationImpl<AudioStream>;

    fn with_speaker<S: fluent_voice_domain::Speaker>(self, speaker: S) -> Self {
        // Convert any Speaker to our concrete SpeakerLine
        let speaker_line = SpeakerLine {
            id: speaker.id().to_string(),
            text: speaker.text().to_string(),
            voice_id: speaker.voice_id().cloned(),
            language: speaker.language().cloned(),
            speed_modifier: speaker.speed_modifier(),
            pitch_range: speaker.pitch_range().cloned(),
        };
        let mut new_lines = self.lines;
        new_lines.push(speaker_line);
        Self {
            lines: new_lines,
            global_language: self.global_language,
            global_speed: self.global_speed,
            model: self.model,
            stability: self.stability,
            similarity: self.similarity,
            speaker_boost: self.speaker_boost,
            style_exaggeration: self.style_exaggeration,
            output_format: self.output_format,
            engine_config: self.engine_config,
            pronunciation_dictionaries: self.pronunciation_dictionaries,
            seed: self.seed,
            previous_text: self.previous_text,
            next_text: self.next_text,
            previous_request_ids: self.previous_request_ids,
            next_request_ids: self.next_request_ids,
            synth_fn: self.synth_fn,
            chunk_processor: self.chunk_processor,
            result_processor: self.result_processor,
        }
    }

    fn language(mut self, lang: Language) -> Self {
        self.global_language = Some(lang);
        self
    }

    fn model(mut self, model: crate::model_id::ModelId) -> Self {
        self.model = Some(model);
        self
    }

    fn stability(mut self, stability: crate::stability::Stability) -> Self {
        self.stability = Some(stability);
        self
    }

    fn similarity(mut self, similarity: crate::similarity::Similarity) -> Self {
        self.similarity = Some(similarity);
        self
    }

    fn speaker_boost(mut self, boost: crate::speaker_boost::SpeakerBoost) -> Self {
        self.speaker_boost = Some(boost);
        self
    }

    fn style_exaggeration(
        mut self,
        exaggeration: crate::style_exaggeration::StyleExaggeration,
    ) -> Self {
        self.style_exaggeration = Some(exaggeration);
        self
    }

    fn output_format(mut self, format: AudioFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    fn pronunciation_dictionary(mut self, dict_id: PronunciationDictId) -> Self {
        self.pronunciation_dictionaries.push(dict_id);
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

    type ChunkBuilder = Self; // For now, same type handles chunk processing

    fn on_chunk<F, T>(mut self, mut processor: F) -> Self::ChunkBuilder
    where
        F: FnMut(Result<T, VoiceError>) -> T + Send + 'static,
        T: Send + 'static,
    {
        // Store the processor function for audio chunk error handling
        self.chunk_processor = Some(Box::new(move |result: Result<Vec<u8>, VoiceError>| -> Vec<u8> {
            // Convert the Vec<u8> result to the generic T type and back
            // This is a simplified implementation - a full implementation would handle proper type conversion
            match result {
                Ok(chunk) => chunk,
                Err(e) => {
                    // Call the user's processor with the error and get the default value
                    let _default_val = processor(Err(e));
                    // Convert back to Vec<u8> - this is simplified
                    Vec::new()
                }
            }
        }));
        self
    }

    fn on_result<F>(mut self, f: F) -> Self
    where
        F: FnMut(VoiceError) -> Vec<u8> + Send + 'static,
    {
        self.result_processor = Some(Box::new(f));
        self
    }

    fn synthesize(self) -> impl Stream<Item = Vec<u8>> + Send + Unpin {
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
            synth_fn: self.synth_fn,
        };
        
        // Convert the i16 stream to bytes stream
        let i16_stream = conversation.into_stream();
        let chunk_size = 1024; // Reasonable chunk size for audio
        
        i16_stream_to_bytes_stream(i16_stream, chunk_size)
    }
}

impl<AudioStream> crate::tts_conversation::TtsConversationChunkBuilder for TtsConversationBuilderImpl<AudioStream>
where
    AudioStream: Stream<Item = i16> + Send + Unpin + 'static,
{
    type Conversation = TtsConversationImpl<AudioStream>;

    fn synthesize(self) -> impl Stream<Item = Vec<u8>> + Send + Unpin {
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
            synth_fn: self.synth_fn,
        };
        
        // Convert the i16 stream to bytes stream
        let i16_stream = conversation.into_stream();
        let chunk_size = 1024; // Reasonable chunk size for audio
        
        i16_stream_to_bytes_stream(i16_stream, chunk_size)
    }
}

/// Convenience module for creating builders
pub mod builder {
    use super::*;

    /// Create a new TTS conversation builder with a custom synthesis function
    pub fn tts_conversation_builder<AudioStream, F>(
        synth_fn: F,
    ) -> TtsConversationBuilderImpl<AudioStream>
    where
        AudioStream: Stream<Item = i16> + Send + Unpin + 'static,
        F: FnOnce(&[SpeakerLine], Option<&Language>) -> AudioStream + Send + 'static,
    {
        TtsConversationBuilderImpl::new(synth_fn)
    }
}
