//! TtsConversationImpl struct and TtsConversation trait implementation

use super::speaker_line::SpeakerLine;
use fluent_voice_domain::{
    audio_format::AudioFormat,
    language::Language,
    pronunciation_dict::{PronunciationDictId, RequestId},
    TtsConversation,
};
use futures::Stream;

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
    pub voice_clone_path: Option<std::path::PathBuf>,
    pub(super) synth_fn: Box<dyn FnOnce(&[SpeakerLine], Option<&Language>) -> AudioStream + Send>,
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
        24000 // Match the 24kHz used throughout synthesis calculations
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
