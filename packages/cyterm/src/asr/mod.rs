#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod audio;
mod decoder;
mod error;
mod iterator;
mod label;
mod multilingual;
mod predict;
mod sample;
mod whisper_loop;

#[cfg(feature = "async")]
mod stream;
mod vad;

use error;
pub use error::Error;
pub use iterator::{IteratorExt, LabelIterator, PredictIterator};
pub use label::LabeledAudio;
pub use sample::Sample;

pub use decoder::WhisperDecoder;
pub use vad::{VoiceActivityDetector, VoiceActivityDetectorBuilder};

#[cfg(feature = "async")]
pub use stream::{LabelStream, PredictStream, StreamExt};
