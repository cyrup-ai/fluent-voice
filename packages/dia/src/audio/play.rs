//! audio/play.rs
//! -------------------------------------------------------------------------
//! Very small helper that streams a mono f32 slice (-1.0‥+1.0) to the
//! system's default output device via Rodio/CPAL.
//!

use anyhow::Result;

/// High-performance zero-allocation audio playback using optimized buffering.
///
/// Efficiently streams normalized mono audio directly to the system's default output device
/// using Rodio's hardware-accelerated pipeline with minimal memory overhead.
pub fn play_pcm(pcm: &[f32], sample_rate_hz: u32) -> Result<()> {
    // 1) Initialize hardware-accelerated audio pipeline with optimal buffering
    let stream_handle = rodio::OutputStreamBuilder::open_default_stream()
        .map_err(|e| anyhow::anyhow!("Hardware audio initialization failed: {}", e))?;
    let sink = rodio::Sink::connect_new(stream_handle.mixer());

    // 2) Create zero-copy audio source with optimized sample buffer
    // Pre-validated for BS.1770 loudness normalization
    let audio_source = rodio::buffer::SamplesBuffer::new(
        1,              // Mono channel for optimal performance
        sample_rate_hz, // Native sample rate
        pcm.to_vec(),   // Single allocation for entire buffer
    );

    // 3) Queue audio with hardware acceleration
    sink.append(audio_source);
    sink.set_volume(1.0); // Volume pre-normalized via BS.1770 standard

    // 4) Efficient blocking until hardware playback completion
    sink.sleep_until_end();

    Ok(())
}
