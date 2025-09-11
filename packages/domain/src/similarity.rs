//! Voice similarity parameter.
//!
//! Controls how closely the synthesized voice matches the original voice.
//! Higher values increase similarity but may affect naturalness.

/// Voice similarity setting between 0.0 and 1.0.
#[derive(Clone, Debug)]
pub struct Similarity(f32);

impl Similarity {
    /// Create a new similarity value.
    ///
    /// Values should be between 0.0 and 1.0, where:
    /// - 0.0 = minimal similarity (more creative/natural voice)
    /// - 1.0 = maximum similarity (very close to original voice)
    ///
    /// Values outside this range will be clamped.
    pub fn new(value: f32) -> Self {
        let clamped = value.clamp(0.0, 1.0);
        Self(clamped)
    }

    /// Get the similarity value.
    pub fn value(&self) -> f32 {
        self.0
    }
}

impl Default for Similarity {
    fn default() -> Self {
        Self(0.75) // Default to high similarity
    }
}
