//! Inclusive pitch range in Hz or MIDI (engine-defined).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchRange {
    /// Lower bound (inclusive).
    pub low: f32,
    /// Upper bound (inclusive).
    pub high: f32,
}

impl PitchRange {
    /// Create new range.
    pub const fn new(low: f32, high: f32) -> Self {
        Self { low, high }
    }
}
