//! Demonstrates the fluent builder API for Text-to-Speech (TTS)

use fluent_voice::prelude::*;
use fluent_voice::synthesize_transform;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    FluentVoice::tts()
        .conversation()
        .with_speaker(
            Speaker::speaker("Narrator")
                .add_line("This is a demonstration of real-time text-to-speech synthesis.")
                .add_line("The audio is generated using the dia-voice engine.")
                .add_line("Thank you for listening!")
                .with_voice("en-US-AriaNeural")
                .with_speed(1.0)
                .build(),
        )
        .synthesize(|conversation| Ok(conversation.into())) // <- Returns AudioStream directly
        .play() // <- This handles all rodio playback internally
        .await; // <- Single await point

    Ok(())
}
