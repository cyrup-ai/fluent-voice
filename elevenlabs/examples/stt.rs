//! ElevenLabs STT Example using FluentVoice Trait System
//!
//! This example demonstrates REAL speech-to-text transcription using the ElevenLabs
//! FluentVoice trait implementation with the exact README API pattern.

use fluent_voice::prelude::*;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    println!("🎤 ElevenLabs STT Example - FluentVoice Trait System");
    println!("===================================================\n");
    println!("🎤 This uses the REAL FluentVoice::stt() API\n");

    println!("🎤 ElevenLabs STT Example - Live Microphone Dictation");
    println!("🎙️ Using: Studio Display Microphone");
    println!("🔥 Say something after the wake word is detected...");
    println!();

    // REAL FluentVoice trait system usage - exact README pattern with microphone
    let transcript = FluentVoice::stt()
        .with_source(SpeechSource::Microphone {
            backend: MicBackend::Named("Studio Display Microphone".to_string()),
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        })
        .vad_mode(VadMode::Accurate)
        .language_hint(Language::ENGLISH_US)
        .word_timestamps(WordTimestamps::On)
        .listen(|result| match result {
            Ok(segment) => Ok(segment.text()), // streaming chunks
            Err(e) => Err(e),
        })
        .collect(); // transcript is now the end-state string

    // Process live microphone dictation
    println!("🔥 FluentVoice STT Live Dictation Starting:");
    println!("===========================================");
    println!("🎙️ Listening to Studio Display Microphone...");
    println!("💬 Speak after the wake word is detected:");
    println!();

    // Process the final transcript result
    match transcript.await {
        Ok(final_transcript) => {
            println!("✅ Transcription Complete!");
            println!("📝 You said: {}", final_transcript);
        }
        Err(e) => {
            eprintln!("❌ Transcription Error: {}", e);
        }
    }

    println!("✅ FluentVoice STT live dictation completed using README pattern!");

    Ok(())
}
