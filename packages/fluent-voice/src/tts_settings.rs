//! Voice synthesis settings.
//!
//! This module provides types for controlling various aspects of the
//! voice synthesis process, including stability, similarity, speaker
//! boost, and style exaggeration.

// Re-export all the individual settings types from domain
pub use fluent_voice_domain::{Similarity, SpeakerBoost, Stability, StyleExaggeration};
