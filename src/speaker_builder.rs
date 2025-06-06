//! fluent_voice/src/speaker_builder.rs
//! -----------------------------------
//! Speaker builder trait definitions

use core::future::Future;

use crate::{
    pitch_range::PitchRange,
    vocal_speed_mod::VocalSpeedMod,
    voice_error::VoiceError,
    voice_timber::VoiceTimber,
};

/* Engine-specific builder implemented in vendor crates */
pub trait SpeakerBuilder: Send {
    type Speaker: crate::speaker::Speaker;

    fn with_speed_modifier(self, speed: VocalSpeedMod) -> Self;
    fn with_pitch_range(self, pitch_range: PitchRange) -> Self;
    fn with_timber(self, timber: VoiceTimber) -> Self;

    fn speak(
        self,
        text: impl Into<String>,
    ) -> impl Future<Output = Result<Self::Speaker, VoiceError>> + Send;
}

/* Static helper: `Speaker::named("Bob") …` */
pub trait SpeakerExt {
    fn named(id: impl Into<String>) -> impl SpeakerBuilder;
}