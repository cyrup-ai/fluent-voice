//! Demonstrates the fluent builder API for Text-to-Speech (TTS)

use fluent_voice::prelude::*;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    // Clean fluent API - all rodio complexity is hidden in .play()
    // Use the concrete synthesize() method that returns AudioStream directly
    FluentVoice::tts()
        .conversation()
        .with_speaker(
            Speaker::speaker("Narrator")
                .with_prelude("Welcome to FluentVoice TTS!")
                .add_line("This is a demonstration of real-time text-to-speech synthesis.")
                .add_line("The audio is generated using the dia-voice engine.")
                .add_line("Thank you for listening!")
                .with_voice("en-US-AriaNeural")
                .with_speed(1.0)
                .build(),
        )
        .synthesize(|conversation| {
            Ok => conversation.into_stream(),  // Returns audio stream
            Err(e) => Err(e),
        })// <- Returns AudioStream directly
        .play() // <- This handles all rodio playback internally
        .await; // Single await point

    Ok(())
}
