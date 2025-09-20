//! Audio processing functions for synthesis task - Part 2

use fluent_voice_domain::VoiceError;
use tokio::sync::mpsc;

pub(super) async fn process_prelude(
    tx: &mpsc::UnboundedSender<Result<fluent_voice_domain::AudioChunk, VoiceError>>,
    prelude_fn: Box<dyn Fn() -> Vec<u8> + Send + 'static>,
    cumulative_time_ms: u64,
) -> u64 {
    let prelude_audio = prelude_fn();

    if !prelude_audio.is_empty() {
        // Calculate duration for prelude audio
        let samples = prelude_audio.len() / 2; // 16-bit PCM
        let duration_ms = (samples as f64 / 24000.0 * 1000.0) as u64;

        let prelude_chunk = fluent_voice_domain::AudioChunk::with_metadata(
            prelude_audio,
            duration_ms,
            0, // start time
            Some("prelude".to_string()),
            Some("Prelude audio".to_string()),
            Some(fluent_voice_domain::AudioFormat::Pcm24Khz),
        );
        let _ = tx.send(Ok(prelude_chunk));

        // Start main synthesis AFTER prelude ends
        duration_ms
    } else {
        cumulative_time_ms
    }
}

pub(super) async fn process_postlude(
    tx: &mpsc::UnboundedSender<Result<fluent_voice_domain::AudioChunk, VoiceError>>,
    postlude_fn: Box<dyn Fn() -> Vec<u8> + Send + 'static>,
    cumulative_time_ms: u64,
) {
    let postlude_audio = postlude_fn();

    if !postlude_audio.is_empty() {
        // Calculate duration for postlude audio
        let samples = postlude_audio.len() / 2; // 16-bit PCM
        let duration_ms = (samples as f64 / 24000.0 * 1000.0) as u64;

        let postlude_chunk = fluent_voice_domain::AudioChunk::with_metadata(
            postlude_audio,
            duration_ms,
            cumulative_time_ms, // Use proper cumulative timing
            Some("postlude".to_string()),
            Some("Postlude audio".to_string()),
            Some(fluent_voice_domain::AudioFormat::Pcm24Khz),
        );
        let _ = tx.send(Ok(postlude_chunk));
    }
}
