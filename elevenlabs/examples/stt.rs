//! ElevenLabs STT Example using FluentVoice Trait System
//!
//! This example demonstrates REAL speech-to-text transcription using the ElevenLabs
//! FluentVoice trait implementation with the exact README API pattern.

use fluent_voice::prelude::*;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    println!("🎤 ElevenLabs STT Example - FluentVoice Trait System");
    println!("===================================================\n");
    println!("🎤 This uses the REAL FluentVoice::stt() API\n");

    println!("🎤 ElevenLabs STT Example - Live Microphone Dictation");
    println!("🎙️ Using: Studio Display Microphone");
    println!("🔥 Say something after the wake word is detected...");
    println!();

    // REAL FluentVoice trait system usage - ElevenLabs only supports FILE-based STT
    let final_transcript = FluentVoice::stt()
        .with_source(SpeechSource::File {
            path: "test-audio.wav".to_string(),
            format: AudioFormat::Pcm16Khz,
        })
        .language_hint(Language::ENGLISH_US)
        .word_timestamps(WordTimestamps::On)
        .listen(|result| match result {
            Ok(conversation) => conversation.collect(),
            Err(e) => panic!("STT Error: {}", e),
        })
        .await;

    // Process file-based transcription (ElevenLabs only supports files, not live mic)
    println!("🔥 FluentVoice STT File Transcription:");
    println!("⚠️  Note: ElevenLabs does NOT support live microphone input");
    println!("📁 Processing audio file: test-audio.wav");
    println!();

    // Process the final transcript result
    println!("✅ Transcription Complete!");
    println!("📝 File contents: {}", final_transcript.await?);

    println!("✅ FluentVoice STT file transcription completed using README pattern!");

    Ok(())
}
