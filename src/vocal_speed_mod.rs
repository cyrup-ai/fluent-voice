//! fluent_voice/src/vocal_speed_mod.rs
//! -----------------------------------
//! Voice speed modifier enum

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocalSpeedMod {
    Slow,
    Normal,
    Fast,
}