//! Canonical STT Example: Full Speech-to-Text Pipeline
//!
//! This example demonstrates the complete STT pipeline with all integrated components:
//! - Wake word detection (koffee)
//! - Voice Activity Detection (VAD)
//! - Speech-to-text transcription (Whisper)
//! - Real-time microphone processing
//! - Speaker diarization and timestamps
//!
//! Features demonstrated:
//! - Real-time microphone input
//! - Wake word activation ("syrup")
//! - Voice activity detection for turn boundaries
//! - High-quality Whisper transcription
//! - Speaker identification and timing
//! - Production-quality error handling
//!
//! Run with: `cargo run --example stt`

use fluent_voice::prelude::*;
use fluent_voice_domain::{
    AudioFormat, Diarization, Language, NoiseReduction, Punctuation, SpeechSource,
    TimestampsGranularity, VadMode, WordTimestamps,
};
use std::error::Error;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🎙️  FluentVoice STT Pipeline Demo");
    println!("=================================");

    // Initialize the STT engine with default providers
    println!("🔧 Initializing STT engine with canonical providers:");
    println!("   • Wake Word Detection: koffee (syrup)");
    println!("   • Voice Activity Detection: fluent_voice_vad");
    println!("   • Speech-to-Text: Whisper (fluent_voice_whisper)");

    // Create STT conversation with full production configuration
    let conversation = FluentVoiceImpl::stt()
        .conversation()
        .with_source(SpeechSource::Microphone {
            device_name: "default".to_string(),
            format: AudioFormat::Pcm16Khz,
        })
        .vad_mode(VadMode::Accurate) // High-accuracy voice activity detection
        .noise_reduction(NoiseReduction::High) // Aggressive background noise filtering
        .language_hint(Language::ENGLISH_US) // Optimize for English (US)
        .diarization(Diarization::On) // Enable speaker identification
        .word_timestamps(WordTimestamps::On) // Generate word-level timestamps
        .timestamps_granularity(TimestampsGranularity::Word) // Word-level timing precision
        .punctuation(Punctuation::On) // Auto-punctuation insertion
        .listen(|conversation| {
            println!("✅ STT conversation configured successfully");
            Ok(conversation.into_stream())
        })
        .await?;

    println!("🎧 Starting real-time speech recognition...");
    println!("💡 Instructions:");
    println!("   1. Say 'syrup' to activate wake word detection");
    println!("   2. Speak clearly after activation");
    println!("   3. The system will transcribe your speech in real-time");
    println!("   4. Press Ctrl+C to stop");
    println!();

    // Process the transcript stream
    let mut transcript_stream = conversation;
    let mut segment_count = 0;
    let mut total_duration = 0.0;
    let mut wake_word_detected = false;

    println!("🔊 Listening for wake word 'syrup'...");

    while let Some(result) = transcript_stream.next().await {
        match result {
            Ok(segment) => {
                segment_count += 1;
                let duration = (segment.end_ms() - segment.start_ms()) as f32 / 1000.0;
                total_duration += duration;

                let text = segment.text();

                // Check if this is a wake word detection
                if text.starts_with("[WAKE WORD:") {
                    wake_word_detected = true;
                    println!("🎯 {} - Wake word detected!", text);
                    println!("🎤 Now listening for speech...");
                    continue;
                }

                // Regular transcription after wake word
                if wake_word_detected {
                    println!("📝 Transcript #{}: {}", segment_count, text);
                    println!(
                        "   ⏱️  Time: {:.2}s - {:.2}s (duration: {:.2}s)",
                        segment.start_ms() as f32 / 1000.0,
                        segment.end_ms() as f32 / 1000.0,
                        duration
                    );

                    if let Some(speaker) = segment.speaker_id() {
                        println!("   👤 Speaker: {}", speaker);
                    }

                    // Show word-level analysis
                    let word_count = text.split_whitespace().count();
                    if word_count > 0 {
                        let words_per_second = word_count as f32 / duration;
                        println!(
                            "   📊 Words: {} ({:.1} words/sec)",
                            word_count, words_per_second
                        );
                    }

                    println!();

                    // Check for exit phrases
                    let lower_text = text.to_lowercase();
                    if lower_text.contains("stop")
                        || lower_text.contains("exit")
                        || lower_text.contains("quit")
                    {
                        println!("🛑 Exit command detected. Stopping transcription...");
                        break;
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Transcription error: {}", e);

                // Handle different error types
                match e {
                    VoiceError::ConfigurationError(_) => {
                        eprintln!("💡 Tip: Check your microphone permissions and device settings");
                    }
                    VoiceError::ProcessingError(_) => {
                        eprintln!("💡 Tip: This might be a temporary audio processing issue");
                        println!("🔄 Continuing to listen...");
                    }
                    _ => {
                        eprintln!("💡 Tip: Unexpected error type, continuing...");
                    }
                }

                println!();
            }
        }
    }

    println!("📊 Final STT Statistics:");
    println!("   • Total segments: {}", segment_count);
    println!("   • Total audio duration: {:.2} seconds", total_duration);
    if segment_count > 0 {
        println!(
            "   • Average segment length: {:.2} seconds",
            total_duration / segment_count as f32
        );
    }
    println!(
        "   • Wake word activation: {}",
        if wake_word_detected { "Yes" } else { "No" }
    );

    println!("🎉 STT demo completed successfully!");

    Ok(())
}
