//! Voice stability parameter.
//!
//! Controls how consistent the voice characteristics remain throughout
//! the generated audio. Higher values increase consistency but may
//! reduce expressiveness.

/// Voice stability setting between 0.0 and 1.0.
#[derive(Clone, Debug)]
pub struct Stability(f32);

impl Stability {
    /// Create a new stability value.
    ///
    /// Values should be between 0.0 and 1.0, where:
    /// - 0.0 = minimal stability (maximum variation/expressiveness)
    /// - 1.0 = maximum stability (consistent voice but less expressive)
    ///
    /// Values outside this range will be clamped.
    pub fn new(value: f32) -> Self {
        let clamped = value.clamp(0.0, 1.0);
        Self(clamped)
    }

    /// Get the stability value.
    pub fn value(&self) -> f32 {
        self.0
    }
}

impl Default for Stability {
    fn default() -> Self {
        Self(0.5) // Default to medium stability
    }
}
