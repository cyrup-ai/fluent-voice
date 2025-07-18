//! Text-to-Speech Example
//!
//! This example demonstrates the exact README.md syntax for TTS operations.
//! Uses the fluent API with conversation() method and exact syntax patterns.
//!
//! Run with: `cargo run --example tts`

use fluent_voice::prelude::*;
use fluent_voice_domain::{VocalSpeedMod, VoiceId};
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
        .on_chunk(|synthesis_chunk| {
            Ok => synthesis_chunk.into(), // Unwrap each audio chunk
            Err(e) => Err(e),
        })
        .synthesize(|conversation| {
            Ok => conversation.into_stream(),  // Returns audio stream
            Err(e) => Err(e),
        })
        .await?;  // Single await point

    // Process audio samples
    while let Some(sample) = audio_stream.next().await {
        // Play sample or save to file
        println!("Audio sample: {}", sample);
    }

    Ok(())
}
