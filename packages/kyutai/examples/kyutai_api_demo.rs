//! Kyutai TTS API Integration Demo
//!
//! This example demonstrates the Kyutai TTS integration with fluent-voice
//! using mock audio generation to avoid CUDA dependency issues.
//!
//! Features:
//! - KyutaiSpeakerLine struct with all required fields
//! - Proper Speaker trait implementation
//! - AudioChunk streaming with metadata
//! - Multi-speaker conversation flow
//!
//! Run with: `cargo run --example kyutai_api_demo`

use fluent_voice_domain::audio_chunk::AudioChunk;
use fluent_voice_domain::{AudioFormat, PitchRange, Speaker, VocalSpeedMod, VoiceId};
use fluent_voice_kyutai::engine::KyutaiSpeakerLine;
use futures_util::stream;
use std::error::Error;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🎭 Kyutai TTS API Integration Demo");
    println!("==================================\n");

    // Create Southpark character speakers with voice cloning parameters
    let speakers = vec![
        KyutaiSpeakerLine {
            text: "Oh my God, they killed Kenny!".to_string(),
            voice_id: Some(VoiceId::new("cartman-voice-clone-2024")),
            language: None,
            speed_modifier: Some(VocalSpeedMod(1.2)),
            pitch_range: Some(PitchRange::new(120.0, 400.0)),
            speaker_id: "Cartman".to_string(),
            speaker_pcm: None,
        },
        KyutaiSpeakerLine {
            text: "You bastards!".to_string(),
            voice_id: Some(VoiceId::new("stan-voice-clone-2024")),
            language: None,
            speed_modifier: Some(VocalSpeedMod(1.0)),
            pitch_range: Some(PitchRange::new(100.0, 350.0)),
            speaker_id: "Stan".to_string(),
            speaker_pcm: None,
        },
        KyutaiSpeakerLine {
            text: "Dude, this is so not cool.".to_string(),
            voice_id: Some(VoiceId::new("kyle-voice-clone-2024")),
            language: None,
            speed_modifier: Some(VocalSpeedMod(0.9)),
            pitch_range: Some(PitchRange::new(110.0, 380.0)),
            speaker_id: "Kyle".to_string(),
            speaker_pcm: None,
        },
        KyutaiSpeakerLine {
            text: "Mmmph mmmph! (You bastards!)".to_string(),
            voice_id: Some(VoiceId::new("kenny-voice-clone-2024")),
            language: None,
            speed_modifier: Some(VocalSpeedMod(0.8)),
            pitch_range: Some(PitchRange::new(80.0, 300.0)),
            speaker_id: "Kenny".to_string(),
            speaker_pcm: None,
        },
    ];

    // Demonstrate Speaker trait implementation
    println!("📝 Speaker Configuration:");
    for speaker in &speakers {
        println!("  • {}: \"{}\"", speaker.id(), speaker.text());
        if let Some(voice_id) = speaker.voice_id() {
            println!("    Voice ID: {}", voice_id.0);
        }
        if let Some(speed) = speaker.speed_modifier() {
            println!("    Speed: {}x", speed.0);
        }
        if let Some(pitch) = speaker.pitch_range() {
            println!("    Pitch Range: {:.1}Hz - {:.1}Hz", pitch.low, pitch.high);
        }
        println!();
    }

    // Generate demonstration audio chunks for API example
    println!("🎵 Generating demonstration audio chunks...");
    let mut demo_audio_chunks = Vec::new();
    let mut cumulative_time_ms = 0u64;

    for speaker in speakers {
        // Simulate audio generation timing
        let text_length = speaker.text().len();
        let duration_ms = (text_length as f64 * 50.0) as u64; // ~50ms per character

        // Create mock PCM data (1024 bytes = ~5ms at 24kHz 16-bit stereo)
        let mock_pcm_data = vec![0u8; 1024];

        let chunk = AudioChunk::with_metadata(
            mock_pcm_data,
            duration_ms,
            cumulative_time_ms,
            Some(speaker.speaker_id.clone()),
            Some(speaker.text.clone()),
            Some(AudioFormat::Pcm24Khz),
        );

        demo_audio_chunks.push(chunk);
        cumulative_time_ms += duration_ms;

        println!("  ✓ {} ({} ms)", speaker.speaker_id, duration_ms);
    }

    // Create audio stream
    let mut audio_stream = Box::pin(stream::iter(demo_audio_chunks));

    // Process audio stream (README.md pattern)
    println!("\n🔊 Processing audio stream...");
    let mut total_chunks = 0;
    let mut total_duration_ms = 0u64;
    let mut total_audio_bytes = 0usize;

    while let Some(chunk) = audio_stream.next().await {
        total_chunks += 1;
        total_duration_ms += chunk.duration_ms();
        total_audio_bytes += chunk.data().len();

        println!(
            "  Chunk {}: {} ({} ms, {} bytes)",
            total_chunks,
            chunk.speaker_id().unwrap_or("Unknown"),
            chunk.duration_ms(),
            chunk.data().len()
        );
    }

    // Summary
    println!("\n📊 Processing Summary:");
    println!("  • Total chunks: {}", total_chunks);
    println!(
        "  • Total duration: {:.2}s",
        total_duration_ms as f64 / 1000.0
    );
    println!("  • Total audio data: {} bytes", total_audio_bytes);
    println!(
        "  • Average chunk size: {} bytes",
        total_audio_bytes / total_chunks
    );

    println!("\n✅ Kyutai TTS API integration demo completed successfully!");
    println!("   Note: This demo uses mock audio data. Real synthesis requires model files.");

    Ok(())
}
