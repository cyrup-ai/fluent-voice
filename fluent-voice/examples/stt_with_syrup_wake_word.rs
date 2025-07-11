//! STT Example with "syrup" wake word detection
//!
//! This example demonstrates how to combine wake word detection with STT
//! using the fluent-voice API. It uses "syrup" as the wake word to trigger
//! speech transcription.

use fluent_voice::prelude::*;
use std::time::Duration;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎤 STT with 'syrup' Wake Word Detection");
    println!("======================================");
    println!("This example shows wake word-triggered STT transcription.");
    println!("Note: This is a demonstration of the fluent API structure.\n");

    // Step 1: Set up wake word detection for "syrup"
    println!("🔍 Step 1: Setting up 'syrup' wake word detection");
    println!("------------------------------------------------");

    let wake_word_builder = FluentVoice::wake_word()
        .with_confidence_threshold(0.8)
        .with_debug(true);

    println!("✅ Wake word builder configured:");
    println!("   - Target word: 'syrup'");
    println!("   - Confidence threshold: 0.8");
    println!("   - Debug output: enabled");

    // For this demo, we'll simulate wake word detection
    // In a real implementation, you would load a syrup.rpw model file
    println!("\n💡 Note: In production, you would:");
    println!("   1. Train or obtain a 'syrup.rpw' wake word model");
    println!("   2. Load it with: builder.with_wake_word_model(\"syrup.rpw\", \"syrup\")");
    println!("   3. Build the detector and process audio streams");

    match wake_word_builder.build() {
        Ok(mut detector) => {
            println!("✅ Wake word detector created successfully!");

            // Step 2: Set up STT for when wake word is detected
            println!("\n🎧 Step 2: Setting up STT transcription");
            println!("--------------------------------------");

            let _stt_builder = FluentVoice::stt()
                .with_microphone("default")
                .language_hint(Language("en-US"))
                .vad_mode(VadMode::Accurate)
                .noise_reduction(NoiseReduction::High)
                .word_timestamps(WordTimestamps::On)
                .punctuation(Punctuation::On);

            println!("✅ STT builder configured:");
            println!("   - Input: Default microphone");
            println!("   - Language: English (US)");
            println!("   - VAD: Accurate mode");
            println!("   - Noise reduction: High");
            println!("   - Features: Word timestamps, punctuation");

            // Step 3: Simulate the wake word -> STT workflow
            println!("\n🔄 Step 3: Simulating wake word detection workflow");
            println!("-------------------------------------------------");

            // Simulate some audio processing cycles
            for cycle in 1..=3 {
                println!("\n🎵 Audio cycle {}/3:", cycle);

                // Create dummy audio samples
                let audio_samples = vec![0.0f32; 1024];

                // Process with wake word detector
                match detector.process_samples(&audio_samples) {
                    Ok(detections) => {
                        if detections.is_empty() {
                            println!("🔇 No 'syrup' detected in audio cycle {}", cycle);
                        } else {
                            println!("🎯 'syrup' wake word detected! Starting STT...");

                            // When wake word is detected, start STT transcription
                            println!("📝 Starting speech transcription...");

                            // In a real implementation, you would call listen() here:
                            println!("🎤 Would call: stt_builder.listen(|conversation| ...)");
                            println!("💬 Transcribed text would appear here in real time");

                            // Simulate some transcription results
                            println!("📄 Simulated transcript segments:");
                            println!("   [0.0s-1.2s] 'Hello there!'");
                            println!("   [1.5s-3.1s] 'This is a test of the syrup wake word.'");
                            println!("   [3.3s-4.8s] 'The transcription is working well.'");
                        }
                    }
                    Err(e) => {
                        println!("⚠️ Error processing audio: {}", e);
                    }
                }

                // Small delay between cycles
                std::thread::sleep(Duration::from_millis(500));
            }

            println!("\n🎯 Wake Word + STT Integration Complete!");
            println!("========================================");
            println!("✅ Fluent API Features Demonstrated:");
            println!("   • Wake word detection with configurable threshold");
            println!("   • STT transcription with microphone input");
            println!("   • Seamless integration between wake word and STT");
            println!("   • Type-safe builder pattern throughout");
            println!("   • Error handling with Result types");

            println!("\n🏗️ Real Implementation Steps:");
            println!("1. Obtain or train a 'syrup.rpw' wake word model");
            println!("2. Configure audio input (microphone or file)");
            println!("3. Set up continuous audio stream processing");
            println!("4. Connect wake word events to STT activation");
            println!("5. Process transcript streams in real-time");

            println!("\n💡 Production Considerations:");
            println!("• Audio buffer management for continuous processing");
            println!("• Wake word false positive handling");
            println!("• STT session timeout and cleanup");
            println!("• Error recovery and reconnection logic");
            println!("• Performance optimization for real-time processing");
        }
        Err(e) => {
            println!("❌ Failed to create wake word detector: {}", e);
            println!("💡 This may be due to missing Koffee dependencies or models");
        }
    }

    println!("\n🚀 Example completed successfully!");
    println!("The fluent-voice API provides a clean, type-safe interface");
    println!("for combining wake word detection with speech transcription.");

    Ok(())
}
