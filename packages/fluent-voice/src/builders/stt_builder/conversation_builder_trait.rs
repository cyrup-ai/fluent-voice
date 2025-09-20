//! SttConversationBuilder trait implementation

use super::conversation_builder_core::SttConversationBuilderImpl;
use super::conversation_impl::SttConversationImpl;
use fluent_voice_domain::{
    language::Language,
    noise_reduction::NoiseReduction,
    speech_source::SpeechSource,
    timestamps::{Diarization, Punctuation, TimestampsGranularity, WordTimestamps},
    transcription::TranscriptionStream,
    vad_mode::VadMode,
};

impl<S> crate::stt_conversation::SttConversationBuilder for SttConversationBuilderImpl<S>
where
    S: TranscriptionStream + 'static,
{
    type Conversation = SttConversationImpl<S>;

    fn with_source(mut self, src: SpeechSource) -> Self {
        self.source = Some(src);
        self
    }

    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.vad_mode = Some(mode);
        self
    }

    fn noise_reduction(mut self, level: NoiseReduction) -> Self {
        self.noise_reduction = Some(level);
        self
    }

    fn language_hint(mut self, lang: Language) -> Self {
        self.language_hint = Some(lang);
        self
    }

    fn diarization(mut self, d: Diarization) -> Self {
        self.diarization = Some(d);
        self
    }

    fn word_timestamps(mut self, w: WordTimestamps) -> Self {
        self.word_timestamps = Some(w);
        self
    }

    fn timestamps_granularity(mut self, g: TimestampsGranularity) -> Self {
        self.timestamps_granularity = Some(g);
        self
    }

    fn punctuation(mut self, p: Punctuation) -> Self {
        self.punctuation = Some(p);
        self
    }

    // on_chunk method removed - now provided by ChunkHandler trait implementation

    fn on_result<F>(self, _f: F) -> Self
    where
        F: FnMut(fluent_voice_domain::VoiceError) -> String + Send + 'static,
    {
        // Store the result processor for error handling
        // For now, we'll just return self until we implement storage
        self
    }

    fn on_wake<F>(self, _f: F) -> Self
    where
        F: FnMut(String) + Send + 'static,
    {
        // Store the wake processor for wake word detection
        // For now, we'll just return self until we implement storage
        self
    }

    fn on_turn_detected<F>(self, _f: F) -> Self
    where
        F: FnMut(Option<String>, String) + Send + 'static,
    {
        // Store the turn detection processor
        // For now, we'll just return self until we implement storage
        self
    }

    fn on_prediction<F>(mut self, f: F) -> Self
    where
        F: FnMut(String, String) + Send + 'static,
    {
        self.prediction_processor = Some(Box::new(f));
        self
    }
}
