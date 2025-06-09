mod builder;
mod microphone;
mod multilingual;
mod pcm_decode;

mod transcript;
mod types;
mod whisper;

pub use transcript::Transcript;
pub use types::TtsChunk;

/// Namespace for the fluent transcription builder.
pub struct Whisper;

impl Whisper {
    /// Begin transcribing an audio file and obtain a builder.
    pub fn transcribe<P: Into<String>>(path: P) -> builder::TranscribeBuilder {
        builder::transcribe(path)
    }
}
