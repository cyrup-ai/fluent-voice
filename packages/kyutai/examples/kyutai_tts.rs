//! Canonical TTS Example: Voice-Cloned Southpark Characters
//!
//! This example demonstrates the exact README.md syntax with voice-cloned Southpark characters
//! using dia voice as the default TTS provider.
//!
//! Features:
//! - Voice-cloned Southpark characters (Kenny, Cartman, Stan, Kyle)
//! - Exact README.md "Ok =>" closure syntax
//! - Dia voice neural TTS synthesis
//! - Multi-speaker conversation flow
//!
//! Run with: `cargo run --example tts`

use fluent_voice::prelude::*;
use fluent_voice_domain::{Language, Speaker, VocalSpeedMod, VoiceId};
use std::error::Error;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🎭 Southpark Voice-Cloned TTS Demo");
    println!("=================================\n");

    // Create TTS conversation with voice-cloned Southpark characters
    // Following exact README.md syntax pattern
    let mut audio_stream = FluentVoice::tts()
        .conversation()
        .with_speaker(
            Speaker::speaker("Kenny")
                .voice_id(VoiceId::new("kenny-voice-clone-2024"))
                .with_speed_modifier(VocalSpeedMod(0.9))
                .speak("Mmmph mmmph mmmph! (Oh my God, they killed Kenny!)")
                .build(),
        )
        .with_speaker(
            Speaker::speaker("Cartman")
                .voice_id(VoiceId::new("cartman-voice-clone-2024"))
                .with_speed_modifier(VocalSpeedMod(1.1))
                .speak("Respect my authoritah! I'm not fat, I'm big-boned!")
                .build(),
        )
        .with_speaker(
            Speaker::speaker("Stan")
                .voice_id(VoiceId::new("stan-voice-clone-2024"))
                .with_speed_modifier(VocalSpeedMod(1.0))
                .speak("Oh my God, this is so messed up. Seriously, you guys.")
                .build(),
        )
        .with_speaker(
            Speaker::speaker("Kyle")
                .voice_id(VoiceId::new("kyle-voice-clone-2024"))
                .with_speed_modifier(VocalSpeedMod(1.05))
                .speak("That's not cool, Cartman! You can't just do that!")
                .build(),
        )
        .with_speaker(
            Speaker::speaker("Kenny")
                .voice_id(VoiceId::new("kenny-voice-clone-2024"))
                .speak("Mmmph mmmph mmmph mmmph! (Yeah, that's really messed up!)")
                .build(),
        )
        .on_chunk(|result| match result {
            Ok(chunk) => chunk,
            Err(e) => {
                eprintln!("TTS error: {}", e);
                AudioChunk::bad_chunk(e.to_string())
            }
        })
        .synthesize();

    // Process audio samples exactly like README.md
    println!("🎵 Processing Southpark character voices...");
    let mut total_samples = 0;
    let mut audio_data = Vec::new();

    while let Some(sample) = audio_stream.next().await {
        // Play sample or save to file (README.md pattern)
        audio_data.push(sample);
        total_samples += 1;

        if total_samples % 10000 == 0 {
            println!("   📊 Processed {} audio samples", total_samples);
        }
    }

    println!("✅ Voice synthesis complete!");
    println!("   🎭 Characters: Kenny, Cartman, Stan, Kyle");
    println!("   📊 Total samples: {}", total_samples);
    println!("   ⏱️  Duration: {:.1}s", total_samples as f32 / 22050.0);
    println!("   🎤 Voice cloning: Enabled");
    println!("   🔊 TTS Engine: dia-voice (default)");

    Ok(())
}
