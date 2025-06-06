//! fluent_voice/src/pitch_range.rs
//! -------------------------------
//! Pitch range value type

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchRange {
    pub low:  f32,
    pub high: f32,
}