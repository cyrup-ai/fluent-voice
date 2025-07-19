//! Demonstrates the fluent builder API for Text-to-Speech (TTS)

use fluent_voice::prelude::*;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    // Clean fluent API - all rodio complexity is hidden in .play()
    FluentVoice::tts()
        .conversation()
        .with_speaker(
            Speaker::speaker("Narrator")
                .voice_id(VoiceId::new("voice-uuid"))
                .with_speed_modifier(VocalSpeedMod(0.9))
                .speak("Hello, world!")
                .build(),
        )
        .with_speaker(
            Speaker::speaker("Bob")
                .with_speed_modifier(VocalSpeedMod(1.1))
                .speak("Hi Alice! How are you today?")
                .build(),
        )
        .synthesize(synthesize_transform!(|conversation| {
            Ok  => conversation.into_stream(),  // Returns audio stream
            Err(e) => panic!("Synthesis error: {:?}", e),
        }))
        .play() // <- This handles all rodio complexity internally
        .await?; // Single await point

    Ok(())
}
