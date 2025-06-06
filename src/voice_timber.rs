//! fluent_voice/src/voice_timber.rs
//! --------------------------------
//! Voice timber quality enum

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceTimber {
    Thin,
    Warm,
    Rich,
}