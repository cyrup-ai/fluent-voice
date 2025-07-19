use fluent_voice_domain::{AudioChunk, VoiceError};
use futures::Stream;
use rodio::Sink;
use std::pin::Pin;

/// A wrapper around an audio stream that provides a fluent `.play()` method
pub struct AudioStream {
    stream: Pin<Box<dyn Stream<Item = AudioChunk> + Send + Unpin>>,
}

impl AudioStream {
    /// Create a new AudioStream wrapper
    pub fn new(stream: Pin<Box<dyn Stream<Item = AudioChunk> + Send + Unpin>>) -> Self {
        Self { stream }
    }

    /// Play the audio stream in real time using rodio
    ///
    /// This method encapsulates all rodio complexity and plays AudioChunk objects
    /// as they arrive from the stream. The user just calls .play() and hears audio.
    pub async fn play(mut self) -> Result<(), VoiceError> {
        use futures::StreamExt;
        use rodio::{OutputStreamBuilder, Sink};

        // Initialize audio output (all rodio complexity hidden here)
        let stream_handle = OutputStreamBuilder::open_default_stream().map_err(|e| {
            VoiceError::Configuration(format!("Failed to initialize audio output: {}", e))
        })?;
        let sink = Sink::connect_new(stream_handle.mixer());

        println!("🎵 Starting real-time audio playback...");

        // Process AudioChunk objects as they arrive
        while let Some(audio_chunk) = self.stream.next().await {
            println!(
                "🔊 Playing audio chunk: {} bytes, duration: {}ms",
                audio_chunk.data.len(),
                audio_chunk.duration_ms
            );

            if let Some(text) = &audio_chunk.text {
                println!("🎤 Speaking: '{}'", text);
            }

            // Play the audio data from this chunk
            self.play_audio_chunk(&sink, &audio_chunk)?;
        }

        // Wait for all audio to finish playing
        sink.sleep_until_end();
        println!("🎵 Audio playback completed!");

        Ok(())
    }

    /// Internal method to play a single AudioChunk (all rodio complexity hidden)
    fn play_audio_chunk(&self, sink: &Sink, audio_chunk: &AudioChunk) -> Result<(), VoiceError> {
        let audio_bytes = &audio_chunk.data;

        if audio_bytes.is_empty() {
            return Ok(());
        }

        // Create audio source from raw bytes
        let cursor = std::io::Cursor::new(audio_bytes.clone());

        // Try to decode as WAV first, then fall back to raw PCM
        match rodio::Decoder::new(cursor.clone()) {
            Ok(decoder) => {
                sink.append(decoder);
            }
            Err(_) => {
                // Fall back to raw 16-bit PCM - convert to f32 samples for rodio
                let samples: Vec<f32> = audio_bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let sample_i16 = i16::from_le_bytes([chunk[0], chunk[1]]);
                        sample_i16 as f32 / 32768.0 // Convert to f32 range [-1.0, 1.0]
                    })
                    .collect();

                let buffer = rodio::buffer::SamplesBuffer::new(1, 24000, samples);
                sink.append(buffer);
            }
        }

        Ok(())
    }
}
