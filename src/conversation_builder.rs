//! fluent_voice/src/conversation_builder.rs
//! ----------------------------------------
//! Conversation builder trait definition

use core::future::Future;

use crate::{
    audio_stream::AudioStream,
    pitch_range::PitchRange,
    vocal_speed_mod::VocalSpeedMod,
    voice_error::VoiceError,
};

/* Engine's fluent conversation builder */
pub trait ConversationBuilder: Sized + Send {
    type PendingSpeaker: Send;

    fn with_speaker(self, speaker: Self::PendingSpeaker) -> Self;
    fn text(self, text: impl Into<String>) -> Self;
    fn speed(self, speed: VocalSpeedMod) -> Self;
    fn pitch(self, range: PitchRange) -> Self;

    fn play<F, R>(self, handler: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<AudioStream, VoiceError>) -> R + Send + 'static,
        R: Send + 'static;
}