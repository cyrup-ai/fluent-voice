//! Demonstrates the fluent builder API for Speech-to-Text (STT)

use fluent_voice::prelude::*;
use fluent_voice_domain::{
    AudioFormat, Diarization, Language, MicBackend, Punctuation, SpeechSource, VadMode,
    WordTimestamps,
};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    let mut transcript_stream = FluentVoice::stt()
        .conversation()
        .with_source(SpeechSource::File {
            path: "path/to/audio.wav".into(),
            format: AudioFormat::Pcm16Khz,
        })
        .language_hint(Language("en-US"))
        .diarization(Diarization::On)
        .word_timestamps(WordTimestamps::On)
        .punctuation(Punctuation::On)
        .on_chunk(on_chunk_transform!(|transcription_chunk| {
            match transcription_chunk {
                Ok(chunk) => Ok(chunk.into()),
                Err(e) => Err(e),
            }
        }))
        .listen(|conv| match conv {
            Ok(conversation) => Ok(conversation.into_stream()),
            Err(e) => Err(e),
        })
        .await;

    // Process transcript segments
    while let Some(result) = transcript_stream.next().await {
        match result {
            Ok(segment) => {
                println!(
                    "[{:.2}s] {}: {}",
                    segment.start_ms() as f32 / 1000.0,
                    segment.speaker_id().unwrap_or("Unknown"),
                    segment.text()
                );
            }
            Err(e) => eprintln!("Recognition error: {}", e),
        }
    }

    Ok(())
}
