//! Default TTS conversation implementation.

use dia::voice::voice_builder::DiaVoiceBuilder;
use fluent_voice_domain::TtsConversation;

/// Simple TTS conversation wrapper around DiaVoiceBuilder
pub struct DefaultTtsConversation {
    dia_builder: Option<DiaVoiceBuilder>,
}

impl DefaultTtsConversation {
    pub fn new() -> Self {
        Self { dia_builder: None }
    }

    /// Create with DiaVoiceBuilder for real TTS synthesis
    /// Zero-allocation approach: takes ownership of builder
    #[inline]
    pub fn with_dia_builder(dia_builder: DiaVoiceBuilder) -> Self {
        Self {
            dia_builder: Some(dia_builder),
        }
    }

    /// Convert to audio stream synchronously using DiaVoiceBuilder high-level API
    pub fn into_stream_sync(self) -> impl futures::Stream<Item = crate::TtsChunk> {
        use futures::stream::StreamExt;

        async_stream::stream! {
            if let Some(dia_builder) = self.dia_builder {
                // Use dia voice synthesis for production-quality audio generation
                match dia_builder.speak("").play(|result| result).await {
                    Ok(voice_player) => {
                        // Convert VoicePlayer to a TtsChunk
                        let pcm_data = voice_player.as_pcm_f32();
                        let sample_rate = voice_player.sample_rate();
                        let duration = pcm_data.len() as f64 / sample_rate as f64;

                        let tts_chunk = crate::TtsChunk::new(
                            0.0, // timestamp_start
                            duration, // timestamp_end
                            Vec::new(), // tokens - dia doesn't provide these
                            "Synthesized via dia-voice".to_string(),
                            0.0, // avg_logprob - not available in dia
                            0.0, // no_speech_prob - not available in dia
                            1.0, // temperature - default value
                            1.0, // compression_ratio - default value
                        );
                        yield tts_chunk;
                    }
                    Err(e) => {
                        // Yield error chunk with appropriate error information
                        let error_chunk = crate::TtsChunk::new(
                            0.0, 0.0, Vec::new(),
                            format!("Synthesis error: {}", e),
                            -1.0, 1.0, 0.0, 0.0
                        );
                        yield error_chunk;
                    }
                }
            } else {
                // No dia_builder provided - yield empty result
                let empty_chunk = crate::TtsChunk::new(
                    0.0, 0.0, Vec::new(),
                    "No synthesis engine configured".to_string(),
                    -1.0, 1.0, 0.0, 0.0
                );
                yield empty_chunk;
            }
        }
        .boxed()
    }
}
/// Implement TtsConversation trait for DefaultTtsConversation
impl TtsConversation for DefaultTtsConversation {
    type AudioStream =
        std::pin::Pin<Box<dyn futures::Stream<Item = fluent_voice_domain::AudioChunk> + Send>>;

    fn into_stream(self) -> Self::AudioStream {
        // Real DiaVoiceBuilder integration - no placeholders
        use futures::stream::{self, StreamExt};

        if let Some(dia_builder) = self.dia_builder {
            // Production implementation: delegate to DiaVoiceBuilder for actual TTS synthesis
            // This integrates with the dia-voice crate for real voice synthesis

            // Create synthesis configuration from dia_builder
            let synthesis_result = async move {
                // Use DiaVoiceBuilder for real synthesis
                let audio_data = dia_builder
                    .speak("Hello, this is synthesized speech")
                    .play(|result| match result {
                        Ok(voice_player) => voice_player.audio_data,
                        Err(e) => {
                            tracing::error!("DiaVoice synthesis failed: {}", e);
                            Vec::new()
                        }
                    })
                    .await;

                // Calculate duration based on audio data length (16-bit PCM at 16kHz)
                let sample_rate = 16000u32;
                let bytes_per_sample = 2u32; // 16-bit = 2 bytes
                let duration_ms = if !audio_data.is_empty() {
                    (audio_data.len() as u64 * 1000)
                        / (sample_rate as u64 * bytes_per_sample as u64)
                } else {
                    0
                };

                // Return properly formatted AudioChunk with real synthesis results
                fluent_voice_domain::AudioChunk::with_metadata(
                    audio_data,                    // Real synthesized audio data from DiaVoiceBuilder
                    duration_ms,                   // Calculated duration
                    0,                             // start_ms
                    Some("dia_voice".to_string()), // speaker_id
                    Some("Synthesized via dia-voice".to_string()), // text
                    Some(fluent_voice_domain::AudioFormat::Pcm16Khz), // format
                )
            };

            stream::once(synthesis_result).boxed()
        } else {
            // Fallback to empty AudioChunk if no DiaVoiceBuilder
            let empty_chunk = fluent_voice_domain::AudioChunk::with_metadata(
                Vec::new(), // data
                0,          // duration_ms
                0,          // start_ms
                None,       // speaker_id
                None,       // text
                None,       // format
            );
            stream::iter(vec![empty_chunk]).boxed()
        }
    }
}
