mod builder;
#[cfg(feature = "microphone")]
mod microphone;
pub mod multilingual;
mod pcm_decode;
mod stream;
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
pub use whisper::{WhichModel, Task, Decoder, DecodingResult, Segment};
#[cfg(not(feature = "microphone"))]
pub use whisper::{Model, token_id};
