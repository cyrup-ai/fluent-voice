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
// Use Sample trait from working VAD implementation
pub use fluent_voice_vad::Sample;

// Re-export whisper types from the working implementation
pub use fluent_voice_whisper::{ModelConfig, WhisperTranscriber, WhisperStream, Transcript};
pub use fluent_voice_whisper::{Decoder as WhisperDecoder, DecodingResult, Segment, Task, WhichModel};
// Re-export VAD types from the working implementation
pub use fluent_voice_vad::{VoiceActivityDetector, VoiceActivityDetectorBuilder};

#[cfg(feature = "async")]
pub use stream::{LabelStream, PredictStream, StreamExt};
