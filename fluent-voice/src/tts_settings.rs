//! Voice synthesis settings.
//!
//! This module provides types for controlling various aspects of the
//! voice synthesis process, including stability, similarity, speaker
//! boost, and style exaggeration.

// Re-export all the individual settings types
pub use crate::similarity::Similarity;
pub use crate::speaker_boost::SpeakerBoost;
pub use crate::stability::Stability;
pub use crate::style_exaggeration::StyleExaggeration;
