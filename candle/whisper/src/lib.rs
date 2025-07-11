mod builder;
mod microphone;
mod multilingual;
mod pcm_decode;
mod stream;
mod transcript;
mod types;
mod whisper;

pub use builder::WhisperEngine;
pub use microphone::{Model, token_id};
pub use stream::WhisperStream;
pub use transcript::Transcript;
pub use types::TtsChunk;

/// Namespace for the fluent transcription builder.
pub struct Whisper;

impl Whisper {
    /// Begin transcribing an audio file and obtain a builder.
    pub fn transcribe<P: Into<String>>(path: P) -> builder::WhisperBuilder {
        builder::transcribe(path)
    }
}
