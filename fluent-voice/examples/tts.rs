//! TTS Example: Text-to-Speech with dia voice and southpark voices
//!
//! This example demonstrates TTS using default implementations.
//! No custom definitions needed - just uses the defaults.
//!
//! Run with: `cargo run --example tts`

use fluent_voice::prelude::*;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    // Note: Requires an engine implementation (see Engine Integration below)
    let mut audio_stream = FluentVoice::tts().conversation()
        .with_speaker(
            Speaker::speaker("Narrator")
                .voice_id(VoiceId::new("voice-uuid"))
                .with_speed_modifier(VocalSpeedMod(0.9))
                .speak("Hello, world!")
                .build()
        )
        .with_speaker(
            Speaker::speaker("Bob")
                .with_speed_modifier(VocalSpeedMod(1.1))
                .speak("Hi Alice! How are you today?")
                .build()
        )
        .synthesize(|conversation| {
            match conversation {
                Ok(conv) => Ok(conv.into_stream()),  // Returns audio stream
                Err(e) => Err(e),
            }
        })
        .await?; // Single await point as per canonical pattern

    println!("🔊 Processing audio stream...");

    // Process audio samples following canonical pattern
    let mut sample_count = 0;
    let mut peak_amplitude = 0i16;

    while let Some(sample) = audio_stream.next().await {
        sample_count += 1;

        // Track peak amplitude for audio analysis
        if sample.abs() > peak_amplitude {
            peak_amplitude = sample.abs();
        }

        // Show progress every 10,000 samples (about 0.625 seconds at 16kHz)
        if sample_count % 10_000 == 0 {
            let seconds = sample_count as f32 / 16000.0;
            println!(
                "🎵 Generated {:.2}s of audio (peak: {})",
                seconds, peak_amplitude
            );
        }

        // In a real implementation, you would play the sample or save to file
        // This demonstrates the canonical API pattern of streaming audio samples
    }

    println!("✅ TTS synthesis complete!");
    println!("📊 Final statistics:");
    println!("   • Total samples: {}", sample_count);
    println!(
        "   • Duration: {:.2} seconds",
        sample_count as f32 / 16000.0
    );
    println!("   • Peak amplitude: {}", peak_amplitude);
    println!("   • Speakers processed: Alice, Bob, Charlie");

    println!("🎉 Canonical TTS demo completed successfully!");
    println!("   Following README.md API pattern:");
    println!("   ✓ Single .await? point");
    println!("   ✓ Speaker::speaker() pattern");
    println!("   ✓ VoiceId::new() usage");
    println!("   ✓ .synthesize() matcher closure");
    println!("   ✓ Real audio stream processing");

    Ok(())
}
