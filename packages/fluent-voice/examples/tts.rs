//! Demonstrates the fluent builder API for Text-to-Speech (TTS)

use fluent_voice::prelude::*;
use fluent_voice_domain::{VocalSpeedMod, VoiceId};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    // Note: This example requires a registered TTS engine to function.
    // The builder constructs a conversation plan, which is then executed by the engine.

    // The `synthesize` method returns a future that resolves to the audio stream.
    // This demonstrates the high-performance, zero-allocation builder pattern.
    let mut audio_stream = FluentVoice::tts()
        .conversation()
        // The `add_line` method now directly accepts a Speaker builder, no `.build()` needed.
        .add_line(
            Speaker::new("Host")
                .speak("Welcome to the show!")
                .voice_id(VoiceId::new("en-US-JennyNeural")),
        )
        .add_line(
            Speaker::new("Guest")
                .speak("It's great to be here.")
                .with_speed(VocalSpeedMod::new(1.2)),
        )
        // The `synthesize` call is elegantly handled with arrow syntax support.
        .synthesize(|conv| Ok(conv.into_stream()))
        .await?;

    // Process the resulting audio stream.
    while let Some(sample) = audio_stream.next().await {
        // In a real application, you would play the audio here.
        // e.g., play_audio(sample)?;
        println!("Received audio chunk of size: {}", sample.len());
    }

    Ok(())
}
