//! Speaker line processing functions - Part 3

use super::speaker_line::SpeakerLine;
use dia::voice::VoicePool;
use fluent_voice_domain::VoiceError;
use std::sync::Arc;
use tokio::sync::mpsc;

pub(super) async fn process_speaker_line(
    tx: &mpsc::UnboundedSender<Result<fluent_voice_domain::AudioChunk, VoiceError>>,
    pool: &Arc<VoicePool>,
    line: SpeakerLine,
    voice_clone_path: Option<&std::path::Path>,
    chunk_index: usize,
    cumulative_time_ms: u64,
) -> u64 {
    // Generate real TTS audio directly from text using dia-voice
    let result =
        super::synthesis::synthesize_speech_internal(pool, &line.id, &line.text, voice_clone_path)
            .await
            .map(|audio_bytes| {
                create_audio_chunk_with_metadata(audio_bytes, line, chunk_index, cumulative_time_ms)
            });

    // Calculate new cumulative time before sending (to avoid moved value)
    let new_cumulative_time = match &result {
        Ok(chunk) => cumulative_time_ms + chunk.duration_ms(),
        Err(_) => cumulative_time_ms,
    };

    // Send Result through channel
    if tx.send(result).is_err() {
        return cumulative_time_ms; // Receiver dropped, stop processing
    }

    new_cumulative_time
}

fn create_audio_chunk_with_metadata(
    audio_bytes: Vec<u8>,
    line: SpeakerLine,
    chunk_index: usize,
    cumulative_time_ms: u64,
) -> fluent_voice_domain::AudioChunk {
    // Create AudioChunk from generated audio bytes
    let duration_ms = if audio_bytes.is_empty() {
        0
    } else {
        // Calculate duration based on audio data size and sample rate
        // Assuming 16-bit PCM at 24kHz sample rate
        let samples = audio_bytes.len() / 2; // 2 bytes per sample for 16-bit
        let duration_secs = samples as f64 / 24000.0; // 24kHz sample rate
        (duration_secs * 1000.0) as u64
    };

    let start_ms = cumulative_time_ms;

    // Create comprehensive timestamp metadata for this chunk
    let mut timestamp_metadata = fluent_voice_domain::TimestampMetadata::new();
    timestamp_metadata.synthesis_metadata.voice_id = line.id.clone();
    timestamp_metadata.synthesis_metadata.text = line.text.clone();
    timestamp_metadata.synthesis_metadata.output_format = "Pcm24Khz".to_string();

    // Add audio chunk timing information
    let audio_chunk_timestamp = fluent_voice_domain::AudioChunkTimestamp {
        chunk_id: chunk_index,
        start_ms,
        end_ms: start_ms + duration_ms,
        text_segment: line.text.clone(),
        speaker_id: Some(line.id.clone()),
        format: "Pcm24Khz".to_string(),
        size_bytes: audio_bytes.len(),
    };
    timestamp_metadata.add_chunk(audio_chunk_timestamp);

    // Finalize metadata
    if let Err(_) = timestamp_metadata.finalize() {
        // Log error but continue - timestamp data is optional
        tracing::warn!(
            "Failed to finalize timestamp metadata for chunk {}",
            chunk_index
        );
    }

    // Attach timestamp metadata to audio chunk in the stream
    fluent_voice_domain::AudioChunk::with_metadata(
        audio_bytes,
        duration_ms,
        start_ms,
        Some(line.id.clone()),
        Some(line.text.clone()),
        Some(fluent_voice_domain::AudioFormat::Pcm24Khz),
    )
    .with_timestamp_metadata(timestamp_metadata)
    .add_metadata("synthesis_time", serde_json::json!(duration_ms))
    .add_metadata("chunk_sequence", serde_json::json!(chunk_index))
}
