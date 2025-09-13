mod builder;
#[cfg(feature = "microphone")]
mod microphone;
mod multilingual;
#[cfg(any(feature = "encodec", feature = "mimi", feature = "snac"))]
mod pcm_decode;
mod stream;
mod transcript;
mod types;
mod whisper;

pub use builder::{ModelConfig, WhisperTranscriber};
#[cfg(feature = "microphone")]
pub use microphone::{Model, token_id};
pub use multilingual::detect_language;
#[cfg(any(feature = "encodec", feature = "mimi", feature = "snac"))]
pub use pcm_decode::pcm_decode;
pub use stream::WhisperStream;
pub use transcript::Transcript;
pub use types::TtsChunk;
pub use whisper::WhichModel;
#[cfg(not(feature = "microphone"))]
pub use whisper::{Model, token_id};
