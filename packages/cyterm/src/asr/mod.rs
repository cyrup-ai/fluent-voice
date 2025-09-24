#![warn(missing_docs)]
#![doc = include_str!("../../README.md")]

mod error;
mod iterator;
mod label;
mod predict;
mod sample;

#[cfg(feature = "async")]
mod stream;
// Remove local vad module - using external vad crate instead

// use error; // Note: already imported by mod error above
pub use error::Error;
pub use iterator::{IteratorExt, LabelIterator, PredictIterator};
pub use label::LabeledAudio;
// Use external Sample trait and local utility functions
pub use fluent_voice_vad::Sample;
pub use sample::{samples_to_f32, samples_to_mono_f32};

// Re-export whisper types from the new public API
pub use fluent_voice_whisper::{WhisperConversation, WhisperSttBuilder};
// Re-export VAD types from the working implementation
pub use fluent_voice_vad::{VoiceActivityDetector, VoiceActivityDetectorBuilder};

#[cfg(feature = "async")]
pub use stream::{LabelStream, PredictStream, StreamExt};
