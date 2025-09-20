//! MicrophoneBuilder trait implementation - Part 2

use super::conversation_impl::SttConversationImpl;
use super::microphone_builder::MicrophoneBuilderImpl;
use fluent_voice_domain::{
    language::Language,
    noise_reduction::NoiseReduction,
    speech_source::SpeechSource,
    timestamps::{Diarization, Punctuation, TimestampsGranularity, WordTimestamps},
    transcription::TranscriptionStream,
    vad_mode::VadMode,
    VoiceError,
};

impl<S> crate::stt_conversation::MicrophoneBuilder for MicrophoneBuilderImpl<S>
where
    S: TranscriptionStream + 'static,
{
    type Conversation = SttConversationImpl<S>;

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

    fn listen<M, ST>(self, matcher: M) -> ST
    where
        M: FnOnce(Result<Self::Conversation, VoiceError>) -> ST + Send + 'static,
        ST: futures_core::Stream<Item = fluent_voice_domain::TranscriptionSegmentImpl>
            + Send
            + Unpin
            + 'static,
    {
        // Use the device string to determine the backend
        let backend = if self.device == "default" || self.device.is_empty() {
            fluent_voice_domain::MicBackend::Default
        } else {
            fluent_voice_domain::MicBackend::Device(self.device)
        };

        let source = Some(SpeechSource::Microphone {
            backend,
            format: fluent_voice_domain::AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        });

        let conversation_result = Ok(SttConversationImpl {
            source,
            vad_mode: self.vad_mode,
            noise_reduction: self.noise_reduction,
            language_hint: self.language_hint,
            diarization: self.diarization,
            word_timestamps: self.word_timestamps,
            timestamps_granularity: self.timestamps_granularity,
            punctuation: self.punctuation,
            stream_fn: self.stream_fn,
        });

        // Apply the matcher closure to the conversation result
        // The matcher contains the JSON syntax transformed by listen! macro
        matcher(conversation_result)
    }
}
