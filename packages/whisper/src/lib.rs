mod builder;
#[cfg(feature = "microphone")]
pub mod microphone;
pub mod multilingual;
mod pcm_decode;
pub mod prelude;
mod stream;
mod token_filtering;
mod transcript;
mod types;
pub mod whisper;

pub use builder::{ModelConfig, WhisperTranscriber};
#[cfg(feature = "microphone")]
pub use microphone::{Model, token_id};
pub use pcm_decode::pcm_decode;
pub use stream::WhisperStream;
pub use transcript::Transcript;
pub use types::TtsChunk;
pub use whisper::{Decoder, DecodingResult, Segment, Task, WhichModel};
#[cfg(not(feature = "microphone"))]
pub use whisper::{Model, token_id};
