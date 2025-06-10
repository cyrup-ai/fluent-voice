//! ElevenLabs expressive controls.

/// Voice stability control (0.0 to 1.0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stability(pub f32);

/// Voice similarity control (0.0 to 1.0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Similarity(pub f32);

/// Style exaggeration control (0.0 to 1.0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StyleExaggeration(pub f32);

/// Speaker boost toggle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeakerBoost(pub bool);
