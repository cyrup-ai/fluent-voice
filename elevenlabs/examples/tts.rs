//! ElevenLabs TTS Example
//!
//! This example demonstrates text-to-speech synthesis using the ElevenLabs
//! implementation of the fluent-voice API.

use fluent_voice_elevenlabs::{FluentVoice, VoiceError};
use futures_util::StreamExt;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    println!("🚀 ElevenLabs Fluent Voice TTS Demo");
    println!("====================================");
    println!();

    // Ensure ELEVENLABS_API_KEY is set
    if std::env::var("ELEVENLABS_API_KEY").is_err() {
        eprintln!("❌ Please set ELEVENLABS_API_KEY environment variable");
        return Ok(());
    }

    // Simple TTS synthesis using fluent-voice API
    println!("🎯 Simple TTS Synthesis:");
    println!("========================");

    let mut audio_stream = FluentVoice::tts()
        .api_key_from_env()?
        .http3_enabled(true)
        .with_speaker(|builder| {
            Ok(builder
                .named("Sarah")
                .speak("Hello! This is ElevenLabs TTS using the fluent-voice API for elegant voice synthesis.")
                .build())
        })?
        .synthesize(|result| result)
        .await?;

    // Collect audio samples (in a real app, you'd play or save these)
    let mut sample_count = 0;
    while let Some(_sample) = audio_stream.next().await {
        sample_count += 1;
    }

    println!("✅ Generated {} audio chunks", sample_count);
    println!();

    // Advanced TTS with voice customization
    println!("🎵 Voice Customization Demo:");
    println!("============================");

    let mut custom_stream = FluentVoice::tts()
        .api_key_from_env()?
        .with_speaker(|builder| {
            Ok(builder
                .named("Brian")
                .speak("Welcome! This demonstrates the fluent-voice API's flexibility.")
                .build())
        })?
        .synthesize(|result| result)
        .await?;

    // Process the audio
    let mut custom_samples = 0;
    while let Some(_sample) = custom_stream.next().await {
        custom_samples += 1;
    }

    println!("✅ Generated {} audio chunks", custom_samples);
    println!();

    // Demonstrating voice variety
    println!("💬 Voice Variety Demo:");
    println!("======================");

    let voices = vec![
        ("Sarah", "I'm Sarah, with a warm and friendly voice."),
        ("Eric", "I'm Eric, with a professional tone."),
        ("Alice", "I'm Alice, perfect for storytelling!"),
        (
            "Charlie",
            "And I'm Charlie, great for casual conversations.",
        ),
    ];

    for (voice_name, text) in &voices {
        print!("🗣️  {} speaking... ", voice_name);
        io::stdout().flush().ok();

        let mut voice_stream = FluentVoice::tts()
            .api_key_from_env()?
            .with_speaker(|builder| Ok(builder.named(*voice_name).speak(*text).build()))?
            .synthesize(|result| result)
            .await?;

        let mut chunks = 0;
        while let Some(_) = voice_stream.next().await {
            chunks += 1;
        }

        println!("✅ {} chunks", chunks);
    }

    println!();

    // STT Example (file transcription)
    println!("📝 Speech-to-Text Demo:");
    println!("======================");

    // Note: You'll need an audio file for this to work
    let audio_file = "sample_audio.wav";

    if std::path::Path::new(audio_file).exists() {
        let transcript = FluentVoice::stt()
            .api_key_from_env()?
            .transcribe(audio_file)?
            .language("en")
            .with_word_timestamps()
            .emit(|result| result)
            .await?;

        println!("Transcript: {}", transcript.text);
    } else {
        println!(
            "ℹ️  No audio file found at '{}' - skipping STT demo",
            audio_file
        );
    }

    println!();

    println!("💡 Key Features of Fluent-Voice API:");
    println!("====================================");
    println!("• 🔗 Unified API for all voice engines");
    println!("• ⚡ Single .await per operation chain");
    println!("• 🎭 Multi-speaker conversations");
    println!("• 🔧 Engine-agnostic design");
    println!("• 📊 Real-time streaming support");
    println!("• 🛡️ Type-safe builder pattern");
    println!();

    println!("💡 Usage Notes:");
    println!("===============");
    println!("• Set ELEVENLABS_API_KEY environment variable");
    println!("• TTS supports multiple voices and speed modifiers");
    println!("• STT supports file transcription with timestamps");
    println!("• All operations use the fluent-voice builder API");
    println!();

    println!(
        "🎉 Example complete! The fluent-voice API makes voice operations elegant and simple."
    );

    Ok(())
}
