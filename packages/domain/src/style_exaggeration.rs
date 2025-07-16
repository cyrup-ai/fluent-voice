//! Style exaggeration parameter.
//!
//! Controls how strongly the voice style and emotions are expressed.
//! Higher values create more dramatic, expressive speech.

/// Style exaggeration setting between 0.0 and 1.0.
#[derive(Clone, Debug)]
pub struct StyleExaggeration(f32);

impl StyleExaggeration {
    /// Create a new style exaggeration value.
    ///
    /// Values should be between 0.0 and 1.0, where:
    /// - 0.0 = minimal exaggeration (subtle expression)
    /// - 1.0 = maximum exaggeration (dramatic expression)
    ///
    /// Values outside this range will be clamped.
    pub fn new(value: f32) -> Self {
        let clamped = value.clamp(0.0, 1.0);
        Self(clamped)
    }

    /// Get the style exaggeration value.
    pub fn value(&self) -> f32 {
        self.0
    }
}

impl Default for StyleExaggeration {
    fn default() -> Self {
        Self(0.3) // Default to moderate exaggeration
    }
}
