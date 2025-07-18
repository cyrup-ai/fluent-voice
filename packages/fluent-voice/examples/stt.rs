//! Speech-to-Text Example
//!
//! This example demonstrates the exact README.md syntax for STT operations.
//! Uses the fluent API with conversation() method and exact syntax patterns.
//!
//! Run with: `cargo run --example stt`

use fluent_voice::prelude::*;
use cyrup_sugars::on_result;
use fluent_voice_domain::{
    AudioFormat, Diarization, Language, MicBackend, Punctuation, SpeechSource,
    TimestampsGranularity, VadMode, WordTimestamps,
};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    let mut transcript_stream = FluentVoice::stt().conversation()
        .with_source(SpeechSource::Microphone {
            backend: MicBackend::Default,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        })
        .vad_mode(VadMode::Accurate)
        .language_hint(Language("en-US"))
        .diarization(Diarization::On)  // Speaker identification
        .word_timestamps(WordTimestamps::On)
        .punctuation(Punctuation::On)
        .on_chunk(on_result!(
            Ok => transcription_chunk.into(), // Unwrap each chunk
            Err(e) => Err(e)
        ))
        .listen(on_result!(
            Ok => segment.text(),  // streaming chunks
            Err(e) => Err(e)
        ))
        .await?;  // Single await point

    // Process transcript segments
    while let Some(result) = transcript_stream.next().await {
        match result {
            Ok(segment) => {
                println!("[{:.2}s] {}: {}",
                    segment.start_ms() as f32 / 1000.0,
                    segment.speaker_id().unwrap_or("Unknown"),
                    segment.text()
                );
            },
            Err(e) => eprintln!("Recognition error: {}", e),
        }
    }

    Ok(())
}
