//! TtsConversationBuilder trait implementation

use super::builder_core::TtsConversationBuilderImpl;
use super::conversation_impl::TtsConversationImpl;
use super::speaker_line::SpeakerLine;
use crate::tts_conversation::TtsConversationBuilder;
use fluent_voice_domain::{
    audio_format::AudioFormat,
    language::Language,
    pronunciation_dict::{PronunciationDictId, RequestId},
    VoiceError,
};
use futures::Stream;

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
            metadata: hashbrown::HashMap::new(),
            vocal_settings: hashbrown::HashMap::new(),
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
    fn with_voice_clone_path(mut self, path: std::path::PathBuf) -> Self {
        self.voice_clone_path = Some(path);
        self
    }

    #[inline]
    fn additional_params<P>(mut self, params: P) -> Self
    where
        P: Into<std::collections::HashMap<String, String>>,
    {
        self.additional_params = params.into();
        self
    }

    #[inline]
    fn metadata<M>(mut self, meta: M) -> Self
    where
        M: Into<std::collections::HashMap<String, String>>,
    {
        self.metadata = meta.into();
        self
    }

    #[inline]
    fn on_chunk<F>(mut self, processor: F) -> Self::ChunkBuilder
    where
        F: FnMut(Result<fluent_voice_domain::AudioChunk, VoiceError>) -> fluent_voice_domain::AudioChunk + Send + 'static,
    {
        // Store the chunk processor - for now just return self as ChunkBuilder
        // TODO: Implement proper chunk processing storage and handling
        self
    }

    #[inline]
    fn on_result<F>(mut self, f: F) -> Self
    where
        F: FnOnce(Result<Self::Conversation, VoiceError>) + Send + 'static,
    {
        self.result_handler = Some(Box::new(f));
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
                voice_clone_path: self.voice_clone_path,
                synth_fn,
            }),
            None => Err(VoiceError::NotSynthesizable(
                "A synthesis function must be provided".to_string(),
            )),
        };

        // Note: result_handler is not called here due to FnOnce constraints
        // The handler would consume the conversation, preventing the matcher from using it

        // Apply the matcher closure to the conversation result
        // The matcher contains the JSON syntax transformed by synthesize! macro
        matcher(conversation_result)
    }
}
