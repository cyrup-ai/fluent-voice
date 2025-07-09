//! Comprehensive API Documentation and Usage Examples
//!
//! This example demonstrates the correct usage of the fluent-voice API,
//! showing how the actual implementation differs from the simplified
//! pseudo-code shown in the README.

use fluent_voice::prelude::*;
use futures_util::StreamExt;

/// Example 1: Basic STT (Speech-to-Text) Pipeline
///
/// This shows the complete flow from microphone input to transcript
async fn stt_example() -> Result<(), VoiceError> {
    println!("=== STT Example ===\n");

    // Step 1: Create STT builder
    let stt_builder = FluentVoice::stt();

    // Step 2: Configure audio source
    let configured = stt_builder
        .with_source(SpeechSource::Microphone {
            backend: MicBackend::Default,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        })
        // Step 3: Configure recognition parameters
        .vad_mode(VadMode::Accurate)
        .language_hint(Language("en-US"))
        .diarization(Diarization::On)
        .word_timestamps(WordTimestamps::On)
        .punctuation(Punctuation::On);

    // Step 4: Execute with matcher closure
    // NOTE: The README shows simplified syntax, but this is the actual API
    let conversation = configured
        .listen(|result| match result {
            Ok(conv) => Ok(conv),
            Err(e) => Err(e),
        })
        .await?;

    // Step 5: Collect the complete transcript
    let transcript = conversation.collect().await?;

    println!("Transcript: {}", transcript);
    Ok(())
}

/// Example 2: Streaming STT with Real-time Processing
async fn streaming_stt_example() -> Result<(), VoiceError> {
    println!("=== Streaming STT Example ===\n");

    // Get a transcript stream for real-time processing
    let mut transcript_stream = FluentVoice::stt()
        .with_microphone("default")
        .vad_mode(VadMode::Fast) // Use fast mode for lower latency
        .language_hint(Language("en-US"))
        .listen(|result| {
            match result {
                Ok(conversation) => conversation.into_stream(),
                Err(_) => futures::stream::empty(), // Return empty stream on error
            }
        });

    // Process segments as they arrive
    while let Some(segment_result) = transcript_stream.next().await {
        match segment_result {
            Ok(segment) => {
                // Access segment properties through the trait
                use fluent_voice::transcript::TranscriptSegment;
                println!(
                    "[{:>6}ms - {:>6}ms] {}: {}",
                    segment.start_ms(),
                    segment.end_ms(),
                    segment.speaker_id().unwrap_or("Unknown"),
                    segment.text()
                );
            }
            Err(e) => eprintln!("Segment error: {:?}", e),
        }
    }

    Ok(())
}

/// Example 3: TTS (Text-to-Speech) Pipeline
async fn tts_example() -> Result<(), VoiceError> {
    println!("=== TTS Example ===\n");

    // Create a mock TTS engine for demonstration
    struct DemoTtsEngine;
    impl TtsEngine for DemoTtsEngine {
        type Conv = TtsConversationBuilderImpl<futures::stream::Iter<std::vec::IntoIter<i16>>>;

        fn conversation(&self) -> Self::Conv {
            tts_conversation_builder(|lines, lang| {
                // In a real implementation, this would generate audio
                println!("TTS Engine Configuration:");
                println!("  Language: {:?}", lang.map(|l| l.0).unwrap_or("default"));
                println!("  Speakers:");
                for line in lines {
                    println!("    {} says: '{}'", line.id, line.text);
                }

                // Return demo audio samples
                futures::stream::iter(vec![0i16; 1000])
            })
        }
    }

    let engine = DemoTtsEngine;

    // Build a multi-speaker conversation
    let mut audio_stream = engine
        .conversation()
        // Configure global settings
        .language(Language("en-US"))
        .model(ModelId::TurboV2_5)
        .stability(Stability::new(0.7))
        .similarity(Similarity::new(0.8))
        // Add speakers
        .with_speaker(
            Speaker::speaker("Alice")
                .voice_id(VoiceId::new("alice-voice-id"))
                .with_speed_modifier(VocalSpeedMod(0.9))
                .speak("Hello! I'm Alice, speaking a bit slowly.")
                .build(),
        )
        .with_speaker(
            Speaker::speaker("Bob")
                .voice_id(VoiceId::new("bob-voice-id"))
                .with_speed_modifier(VocalSpeedMod(1.1))
                .with_pitch_range(PitchRange::new(0.8, 1.2))
                .speak("Hi Alice! I'm Bob, speaking faster with varied pitch.")
                .build(),
        )
        // Execute synthesis
        .synthesize(|result| match result {
            Ok(conversation) => conversation.into_stream(),
            Err(e) => panic!("Synthesis failed: {:?}", e),
        })
        .await;

    // Process audio samples
    let mut sample_count = 0;
    while let Some(_sample) = audio_stream.next().await {
        sample_count += 1;
        if sample_count >= 100 {
            break;
        } // Just process first 100 samples
    }

    println!("\nProcessed {} audio samples", sample_count);
    Ok(())
}

/// Example 4: File Transcription
async fn file_transcription_example() -> Result<(), VoiceError> {
    println!("=== File Transcription Example ===\n");

    // Transcribe an audio file
    let transcript = FluentVoice::stt()
        .transcribe("path/to/audio.wav")
        .language_hint(Language("en-US"))
        .with_progress("Transcribing: {percent}%")
        .collect()
        .await?;

    println!("File transcript: {}", transcript);
    Ok(())
}

/// Example 5: Working with Value Types
fn value_types_example() {
    println!("=== Value Types Example ===\n");

    // Language codes (BCP-47)
    let english = Language("en-US");
    let spanish = Language("es-ES");
    let japanese = Language("ja-JP");

    // Audio formats
    let format = AudioFormat::Pcm16Khz;

    // Voice settings
    let voice_id = VoiceId::new("unique-voice-identifier");
    let speed = VocalSpeedMod(1.0); // Normal speed
    let pitch = PitchRange::new(0.9, 1.1); // Slight variation

    // TTS parameters
    let stability = Stability::new(0.5); // Balanced
    let similarity = Similarity::new(0.75); // High similarity
    let boost = SpeakerBoost::new(true); // Enhanced separation
    let style = StyleExaggeration::new(0.3); // Moderate

    // STT parameters
    let vad = VadMode::Accurate;
    let noise = NoiseReduction::High;
    let diarization = Diarization::On;
    let timestamps = WordTimestamps::On;
    let punctuation = Punctuation::On;

    println!("All value types created successfully!");
}

/// Example 6: Error Handling
async fn error_handling_example() -> Result<(), VoiceError> {
    println!("=== Error Handling Example ===\n");

    // Example of handling errors in the matcher closure
    let result = FluentVoice::stt()
        .with_microphone("nonexistent-device")
        .listen(|result| match result {
            Ok(conversation) => {
                println!("Conversation started successfully");
                Ok(conversation)
            }
            Err(e) => {
                println!("Failed to start conversation: {:?}", e);
                Err(e)
            }
        })
        .await;

    match result {
        Ok(conversation) => {
            let transcript = conversation.collect().await?;
            println!("Transcript: {}", transcript);
        }
        Err(e) => {
            println!("Error handled gracefully: {:?}", e);
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fluent Voice API Documentation Examples\n");
    println!("This demonstrates the actual API, not the pseudo-code from README\n");

    // Run examples (commented out to avoid runtime errors without real engines)
    // stt_example().await?;
    // streaming_stt_example().await?;
    // tts_example().await?;
    // file_transcription_example().await?;

    value_types_example();
    // error_handling_example().await?;

    println!("\n=== Key API Patterns ===\n");
    println!("1. All operations follow: Builder → Configure → Execute → Await");
    println!("2. The matcher closure receives Result<T, VoiceError>");
    println!("3. You must handle both Ok and Err cases");
    println!("4. The 'Ok =>' syntax in README is pseudo-code");
    println!("5. Use trait imports to access methods (e.g., TranscriptSegment)");

    Ok(())
}
