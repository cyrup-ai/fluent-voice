//! Canonical STT Example: Wake on "syrup", Dictate to Console
//!
//! This example demonstrates the exact README.md syntax with specific behavior:
//! - Wake on "syrup" keyword (koffee wake word detection)
//! - Real-time dictation to console
//! - Turn detection printing (VAD boundaries)
//! - Unwake on "syrup stop" command
//! - Uses default engine implementations (VAD, koffee, whisper)
//!
//! Features:
//! - Exact README.md "Ok =>" closure syntax
//! - Default STT providers (no special settings required)
//! - Real-time microphone transcription
//! - Wake word activation and deactivation
//!
//! Run with: `cargo run --example stt`

use fluent_voice::prelude::*;
use fluent_voice_domain::{
    AudioFormat, Diarization, Language, MicBackend, Punctuation, SpeechSource,
    TimestampsGranularity, VadMode, WordTimestamps,
};
use std::error::Error;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🎙️  Syrup Wake Word STT Demo");
    println!("============================\n");

    // Create STT conversation using exact README.md syntax with enhanced JSON configuration
    // Default engines: VAD, koffee wake word, whisper STT with JSON configuration
    let mut transcript_stream = FluentVoice::stt()
        .conversation()
        .engine_config({"provider" => "whisper", "model" => "large-v3"})
        .with_source(SpeechSource::Microphone {
            backend: MicBackend::Default,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        })
        .vad_mode(VadMode::Accurate)
        .language_hint(Language::ENGLISH_US)
        .diarization(Diarization::On) // Speaker identification
        .word_timestamps(WordTimestamps::On)
        .punctuation(Punctuation::On)
        .on_chunk(|transcription_chunk| {
            Ok => transcription_chunk.into(), // Unwrap each chunk
            Err(e) => Err(e),
        })
        .listen(|conversation| conversation.into_stream()) // Returns transcript stream
        .await?; // Single await point

    println!("🔊 Listening for wake word 'syrup'...");
    println!("💬 Say 'syrup' to start dictation, 'syrup stop' to end\n");

    let mut is_awake = false;
    let mut turn_count = 0;

    while let Some(transcript_result) = transcript_stream.next().await {
        match transcript_result {
            Ok(segment) => {
                let text = segment.text().to_lowercase();

                // Wake word detection
                if !is_awake && text.contains("syrup") {
                    is_awake = true;
                    println!("🟢 WAKE WORD DETECTED: 'syrup'");
                    println!("🎤 Dictation ACTIVE - speak now...\n");
                    continue;
                }

                // Stop command detection
                if is_awake && text.contains("syrup stop") {
                    is_awake = false;
                    println!("🔴 STOP COMMAND: 'syrup stop'");
                    println!("😴 Dictation INACTIVE - say 'syrup' to wake\n");
                    continue;
                }

                // Process dictation when awake
                if is_awake {
                    turn_count += 1;

                    // Turn detection (VAD boundary)
                    println!("🔄 TURN {} DETECTED (VAD boundary)", turn_count);

                    // Real-time dictation to console
                    println!("📝 DICTATION: {}", segment.text());

                    // Show timing and speaker info if available
                    if let Some(speaker) = segment.speaker() {
                        println!("👤 Speaker: {}", speaker);
                    }
                    println!(
                        "⏱️  Duration: {:.2}s\n",
                        (segment.end_ms() - segment.start_ms()) as f32 / 1000.0
                    );
                }
            }
            Err(e) => {
                println!("❌ Transcription error: {}", e);
            }
        }
    }

    println!("\n✅ STT session ended");
    Ok(())
}
