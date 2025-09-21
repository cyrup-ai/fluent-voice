//! ElevenLabs TTS Example with Real Audio Output
//!
//! This example demonstrates text-to-speech synthesis using the FluentVoice API
//! with actual audio output capabilities:
//! - WAV file saving using hound library
//! - Direct audio playback using rodio
//! - Multiple output format options
//!
//! Usage:
//! - `cargo run --example elevenlabs_tts save` - Save to WAV file
//! - `cargo run --example elevenlabs_tts play` - Play directly  
//! - `cargo run --example elevenlabs_tts both` - Save and play

use bytes::Bytes;
use fluent_voice_elevenlabs::{FluentVoice, Result};
use futures_util::StreamExt;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::env;

/// Save audio stream to WAV file using existing hound dependency
async fn save_audio_stream_to_wav(
    mut stream: impl futures_util::Stream<Item = Result<Bytes>> + Unpin,
    path: &str,
) -> Result<()> {
    // ElevenLabs default audio specifications
    let spec = WavSpec {
        channels: 1,         // Mono
        sample_rate: 22050,  // ElevenLabs default
        bits_per_sample: 16, // 16-bit PCM
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;
    let mut total_samples = 0;

    println!("🎵 Saving audio to: {}", path);

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;

        // Convert bytes to i16 samples (assuming 16-bit PCM)
        let samples: Vec<i16> = chunk
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .collect();

        for sample in samples {
            writer.write_sample(sample)?;
            total_samples += 1;
        }
    }

    writer.finalize()?;
    println!("✅ Saved {} samples to {}", total_samples, path);
    Ok(())
}

/// Play audio stream directly using existing playback infrastructure
async fn play_audio_stream(
    audio_stream: impl futures_util::Stream<Item = Result<Bytes>> + Unpin,
) -> Result<()> {
    println!("🔊 Playing audio...");

    // Use existing stream_audio function from utils/playback.rs
    crate::utils::stream_audio(audio_stream).await?;

    println!("✅ Audio playback completed");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 ElevenLabs Fluent Voice TTS Demo");
    println!("====================================");
    println!();

    // Ensure ELEVENLABS_API_KEY is set
    if std::env::var("ELEVENLABS_API_KEY").is_err() {
        eprintln!("❌ Please set ELEVENLABS_API_KEY environment variable");
        return Ok(());
    }

    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("save");

    // Simple TTS synthesis using FluentVoice API
    println!("🎯 Simple TTS Synthesis:");
    println!("========================");

    let audio_stream = FluentVoice::tts()
        .api_key_from_env()?
        .http3_enabled(true)
        .with_speaker(|speaker_builder| {
            Ok(speaker_builder
                .named("Sarah")
                .speak("Hello! This is ElevenLabs TTS using the FluentVoice API.")
                .build())
        })?
        .synthesize(|result| result)
        .await?;

    // Convert to audio-only stream for processing
    let audio_only_stream = audio_stream.audio_only();

    match mode {
        "play" => {
            println!("🔊 Playing audio directly...");
            play_audio_stream(audio_only_stream).await?;
        }
        "save" => {
            println!("💾 Saving audio to file...");
            let output_path = "elevenlabs_sarah_demo.wav";
            save_audio_stream_to_wav(audio_only_stream, output_path).await?;
            println!("✅ Audio saved to: {}", output_path);
        }
        "both" => {
            // Save first, then play the saved file
            println!("💾 Saving audio...");
            let output_path = "elevenlabs_sarah_demo.wav";
            save_audio_stream_to_wav(audio_only_stream, output_path).await?;

            println!("🔊 Playing saved audio...");
            // Use existing playback infrastructure consistently
            let file_data = std::fs::read(output_path)?;
            let bytes_stream = futures_util::stream::once(async { Ok(Bytes::from(file_data)) });
            crate::utils::stream_audio(bytes_stream).await?;
        }
        _ => {
            println!("Usage: cargo run --example elevenlabs_tts [save|play|both]");
            println!("Default: save");
            let output_path = "elevenlabs_sarah_demo.wav";
            save_audio_stream_to_wav(audio_only_stream, output_path).await?;
            println!("✅ Audio saved to: {}", output_path);
        }
    }

    println!();

    // Advanced TTS with voice customization
    println!("🎵 Voice Customization Demo:");
    println!("=============================");

    let custom_stream = FluentVoice::tts()
        .api_key_from_env()?
        .http3_enabled(true)
        .with_speaker(|speaker_builder| {
            Ok(speaker_builder
                .named("Brian")
                .speak("Welcome! This demonstrates the FluentVoice API's flexibility.")
                .build())
        })?
        .synthesize(|result| result)
        .await?;

    // Convert to audio-only stream and save
    let custom_audio_stream = custom_stream.audio_only();
    let custom_output_path = "elevenlabs_brian_demo.wav";
    save_audio_stream_to_wav(custom_audio_stream, custom_output_path).await?;
    println!("✅ Audio saved to: {}", custom_output_path);
    println!();

    // Multi-speaker conversation with multiple synthesis calls
    println!("💬 Multi-Speaker Conversation:");
    println!("==============================");

    // Synthesize Alice's part
    let alice_stream = FluentVoice::tts()
        .api_key_from_env()?
        .http3_enabled(true)
        .with_speaker(|speaker_builder| {
            Ok(speaker_builder
                .named("Alice")
                .speak("Hello everyone! I'm Alice, and I'm excited to demonstrate multi-speaker conversations.")
                .build())
        })?
        .synthesize(|result| result)
        .await?;

    // Convert to audio-only stream and save
    let alice_audio_stream = alice_stream.audio_only();
    let alice_output_path = "elevenlabs_alice_demo.wav";
    save_audio_stream_to_wav(alice_audio_stream, alice_output_path).await?;
    println!("✅ Alice's audio saved to: {}", alice_output_path);

    // Synthesize Bob's part
    let bob_stream = FluentVoice::tts()
        .api_key_from_env()?
        .http3_enabled(true)
        .with_speaker(|speaker_builder| {
            Ok(speaker_builder
                .named("Brian") // Using Brian as Bob isn't available
                .speak("Hi Alice! I'm Bob. This shows how we can have multiple speakers with FluentVoice.")
                .build())
        })?
        .synthesize(|result| result)
        .await?;

    // Convert to audio-only stream and save
    let bob_audio_stream = bob_stream.audio_only();
    let bob_output_path = "elevenlabs_bob_demo.wav";
    save_audio_stream_to_wav(bob_audio_stream, bob_output_path).await?;
    println!("✅ Bob's audio saved to: {}", bob_output_path);

    println!();
    println!("💡 Key Features of FluentVoice TTS API:");
    println!("======================================");
    println!("• 🔗 FluentVoice::tts().with_speaker() pattern");
    println!("• ⚡ HTTP/3 QUIC support for optimal performance");
    println!("• 🎭 Multi-speaker support with separate synthesis calls");
    println!("• 🔧 Engine-agnostic design with trait system");
    println!("• 📊 Real-time streaming audio generation");
    println!("• 🛡️ Type-safe builder pattern with error handling");
    println!("• 🎵 WAV file saving using hound library");
    println!("• 🔊 Direct audio playback using rodio");
    println!();

    println!("💡 Usage Notes:");
    println!("===============");
    println!("• Set ELEVENLABS_API_KEY environment variable");
    println!("• Uses FluentVoice::tts().with_speaker() API pattern");
    println!("• Each speaker requires separate synthesis call");
    println!("• API key loaded from environment with .api_key_from_env()");
    println!("• Audio files saved as 22050 Hz, 16-bit PCM, mono WAV");
    println!();

    println!(
        "🎉 Example complete! The FluentVoice TTS API provides clean, type-safe voice synthesis with real audio output."
    );

    Ok(())
}
