//! Builder module public API

use super::builder_core::TtsConversationBuilderImpl;
use super::speaker_line::SpeakerLine;
use fluent_voice_domain::language::Language;
use futures::Stream;

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
