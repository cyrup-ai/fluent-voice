//! Demonstrates the fluent builder API for Speech-to-Text (STT)

use fluent_voice::prelude::*;
use fluent_voice_domain::{
    transcription::{MessageChunk, TranscriptionSegmentImpl},
    AudioFormat, Diarization, Language, Punctuation, SpeechSource, WordTimestamps,
};
use futures_util::StreamExt;
use cyrup_sugars::prelude::ChunkHandler;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    let transcription = FluentVoice::stt()
        .conversation()
        .with_source(SpeechSource::File {
            path: "path/to/audio.wav".into(),
            format: AudioFormat::Pcm16Khz,
        })
        .language_hint(Language("en-US"))
        .diarization(Diarization::On)
        .word_timestamps(WordTimestamps::On)
        .punctuation(Punctuation::On)
        .on_chunk(|result| match result {
            Ok(segment) => segment,
            Err(e) => TranscriptionSegmentImpl::bad_chunk(e.to_string()),
        })
        .listen(|conversation| match conversation {
            Ok(conv) => conv.into_stream(),
            Err(e) => panic!("Failed to create conversation: {}", e),
        })
        .collect::<Vec<_>>()
        .await;

    println!("Transcription segments: {}", transcription.len());
    Ok(())
}
