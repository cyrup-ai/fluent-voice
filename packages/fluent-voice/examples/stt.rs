//! Demonstrates the fluent builder API for Speech-to-Text (STT)

use fluent_voice::prelude::*;
use fluent_voice_domain::{
    AudioFormat, Diarization, Language, MicBackend, Punctuation, SpeechSource, VadMode,
    WordTimestamps,
};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    // Note: This example requires a registered STT engine and a valid audio file.
    // The builder constructs a recognition plan, which is then executed by the engine.
    let mut transcript_stream = FluentVoice::stt()
        .conversation()
        .with_source(SpeechSource::File {
            path: "path/to/audio.wav".into(),
            format: AudioFormat::Pcm16Khz,
        })
        // Use type-safe constants for better readability and compile-time checks.
        .language_hint(Language::ENGLISH_US)
        .diarization(Diarization::On)
        .word_timestamps(WordTimestamps::On)
        .punctuation(Punctuation::On)
        // The `listen` method returns a future that resolves to the transcript stream.
        // This showcases the elegant, non-blocking API design.
        .listen(|conv| Ok(conv.into_stream()))
        .await?;

    // Process the resulting transcript stream.
    while let Some(result) = transcript_stream.next().await {
        match result {
            Ok(segment) => {
                println!(
                    "[{:.2}s] {}: {}",
                    segment.start_ms() as f32 / 1000.0,
                    // Use a default value for speaker ID if not present.
                    segment.speaker_id().unwrap_or("Unknown"),
                    segment.text()
                );
            }
            Err(e) => eprintln!("Recognition error: {}", e),
        }
    }

    Ok(())
}
