//! Demonstrates the fluent builder API for Speech-to-Text (STT) with microphone listening using ElevenLabs

use fluent_voice_elevenlabs::prelude::*;
use fluent_voice_domain::transcription::TranscriptionSegmentImpl;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    println!("Starting microphone listening for speech-to-text...");
    println!("Speak into your microphone. Press Ctrl+C to stop.");


    // Complete fluent chain with microphone support, VAD and wake word detection
    let _transcript = FluentVoice::stt()
        .with_source(SpeechSource::Microphone {
            backend: MicBackend::Default,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        })
        .vad_mode(VadMode::Accurate)
        .language_hint(Language("en-US"))
        .diarization(Diarization::On)
        .word_timestamps(WordTimestamps::On)
        .punctuation(Punctuation::On)
        .on_chunk(|result| match result {
            Ok(segment) => {
                println!("🎤 ElevenLabs STT: {}", segment.text());
                segment
            },
            Err(_e) => TranscriptionSegmentImpl::new("".to_string(), 0, 0, None),
        })
        .listen(|conversation| match conversation {
            Ok(conv) => {
                conv.into_stream().map(|result| {
                    result.map(|segment| TranscriptionSegmentImpl::new(
                        segment.text().to_string(),
                        segment.start_ms(),
                        segment.end_ms(),
                        segment.speaker_id().map(|s| s.to_string()),
                    ))
                }).boxed()
            },
            Err(_e) => {
                eprintln!("❌ STT conversation error: {}", _e);
                panic!("STT failed: {}", _e);
            }
        }); 

    println!("✅ Microphone listening completed");
    Ok(())
}
