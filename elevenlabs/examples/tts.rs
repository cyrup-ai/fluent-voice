//! ElevenLabs TTS Example using FluentVoice API
//!
//! This example demonstrates text-to-speech synthesis using the actual
//! FluentVoice API with with_speaker() method.

use fluent_voice_elevenlabs::FluentVoice;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ElevenLabs Fluent Voice TTS Demo");
    println!("====================================");
    println!();

    // Ensure ELEVENLABS_API_KEY is set
    if std::env::var("ELEVENLABS_API_KEY").is_err() {
        eprintln!("❌ Please set ELEVENLABS_API_KEY environment variable");
        return Ok(());
    }

    // Simple TTS synthesis using FluentVoice API
    println!("🎯 Simple TTS Synthesis:");
    println!("========================");

    let mut audio_stream = FluentVoice::tts()
        .api_key_from_env()?
        .http3_enabled(true)
        .with_speaker(|speaker_builder| {
            speaker_builder
                .named("Sarah")
                .speak("Hello! This is ElevenLabs TTS using the FluentVoice API.")
                .build()
                .synthesize(|result| result)
        })?;

    // Collect audio samples (in a real app, you'd play or save these)
    let mut sample_count = 0;
    while let Some(_sample) = audio_stream.next().await {
        sample_count += 1;
    }

    println!("✅ Generated {} audio chunks", sample_count);
    println!();

    // Advanced TTS with voice customization
    println!("🎵 Voice Customization Demo:");
    println!("=============================");

    let mut custom_stream = FluentVoice::tts()
        .api_key_from_env()?
        .http3_enabled(true)
        .with_speaker(|speaker_builder| {
            speaker_builder
                .named("Brian")
                .speak("Welcome! This demonstrates the FluentVoice API's flexibility.")
                .build()
                .synthesize(|result| result)
        })?;

    // Process the audio
    let mut custom_samples = 0;
    while let Some(_sample) = custom_stream.next().await {
        custom_samples += 1;
    }

    println!("✅ Generated {} audio chunks", custom_samples);
    println!();

    // Multi-speaker conversation with multiple synthesis calls
    println!("💬 Multi-Speaker Conversation:");
    println!("==============================");

    // Synthesize Alice's part
    let mut alice_stream = FluentVoice::tts()
        .api_key_from_env()?
        .http3_enabled(true)
        .with_speaker(|speaker_builder| {
            speaker_builder
                .named("Alice")
                .speak("Hello everyone! I'm Alice, and I'm excited to demonstrate multi-speaker conversations.")
                .build()
                .synthesize(|result| result)
        })?;
    let mut alice_chunks = 0;
    while let Some(_sample) = alice_stream.next().await {
        alice_chunks += 1;
    }

    // Synthesize Bob's part
    let mut bob_stream = FluentVoice::tts()
        .api_key_from_env()?
        .http3_enabled(true)
        .with_speaker(|speaker_builder| {
            speaker_builder
                .named("Brian") // Using Brian as Bob isn't available
                .speak("Hi Alice! I'm Bob. This shows how we can have multiple speakers with FluentVoice.")
                .build()
                .synthesize(|result| result)
        })?;
    let mut bob_chunks = 0;
    while let Some(_sample) = bob_stream.next().await {
        bob_chunks += 1;
    }

    let total_chunks = alice_chunks + bob_chunks;
    println!(
        "✅ Generated {} audio chunks from multi-speaker conversation (Alice: {}, Bob: {})",
        total_chunks, alice_chunks, bob_chunks
    );
    println!();

    println!("💡 Key Features of FluentVoice TTS API:");
    println!("======================================");
    println!("• 🔗 FluentVoice::tts().with_speaker() pattern");
    println!("• ⚡ HTTP/3 QUIC support for optimal performance");
    println!("• 🎭 Multi-speaker support with separate synthesis calls");
    println!("• 🔧 Engine-agnostic design with trait system");
    println!("• 📊 Real-time streaming audio generation");
    println!("• 🛡️ Type-safe builder pattern with error handling");
    println!();

    println!("💡 Usage Notes:");
    println!("===============");
    println!("• Set ELEVENLABS_API_KEY environment variable");
    println!("• Uses FluentVoice::tts().with_speaker() API pattern");
    println!("• Each speaker requires separate synthesis call");
    println!("• API key loaded from environment with .api_key_from_env()");
    println!();

    println!(
        "🎉 Example complete! The FluentVoice TTS API provides clean, type-safe voice synthesis."
    );

    Ok(())
}
