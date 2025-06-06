//! fluent_voice/src/voice_player.rs
//! --------------------------------
//! Voice player trait definition

use core::future::Future;
use std::path::Path;

use crate::voice_error::VoiceError;

/* Returned by engines for post-playback actions */
pub trait VoicePlayer: Send + Sync {
    fn play(&self) -> impl Future<Output = Result<(), VoiceError>> + Send + '_;
    fn save(
        &self,
        path: impl AsRef<Path> + Send + 'static,
    ) -> impl Future<Output = Result<(), VoiceError>> + Send + '_;
    fn to_bytes(&self) -> impl Future<Output = Result<Vec<u8>, VoiceError>> + Send + '_;

    fn sample_rate(&self) -> u32;
    fn channels(&self) -> u16;
}