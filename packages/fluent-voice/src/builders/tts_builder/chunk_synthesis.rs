//! Synthesis task processing for TtsConversationChunkBuilder - Part 1

use super::speaker_line::SpeakerLine;
use candle_core::Device;
use dia::voice::VoicePool;
use fluent_voice_domain::VoiceError;
use std::sync::Arc;
use tokio::sync::mpsc;

pub(super) async fn process_synthesis_task(
    tx: mpsc::UnboundedSender<Result<fluent_voice_domain::AudioChunk, VoiceError>>,
    lines: Vec<SpeakerLine>,
    voice_clone_path: Option<std::path::PathBuf>,
    prelude: Option<Box<dyn Fn() -> Vec<u8> + Send + 'static>>,
    postlude: Option<Box<dyn Fn() -> Vec<u8> + Send + 'static>>,
    engine_config: Option<hashbrown::HashMap<String, String>>,
) {
    // Initialize voice pool with engine config if provided
    let cache_dir = std::env::temp_dir().join("fluent_voice_cache");
    let device = if let Some(ref config) = engine_config {
        // Apply engine configuration to device selection
        match config.get("device").map(|s| s.as_str()) {
            Some("cuda") => Device::new_cuda(0).unwrap_or(Device::Cpu),
            Some("metal") => Device::new_metal(0).unwrap_or(Device::Cpu),
            _ => Device::Cpu,
        }
    } else {
        Device::Cpu
    };

    let pool = match VoicePool::new_with_config(cache_dir, device) {
        Ok(pool) => Arc::new(pool),
        Err(e) => {
            // Send error result for pool creation failure
            let _ = tx.send(Err(VoiceError::Configuration(format!(
                "Failed to create voice pool: {}",
                e
            ))));
            return;
        }
    };

    // Execute prelude function if provided
    let mut cumulative_time_ms = 0u64;
    if let Some(prelude_fn) = prelude {
        cumulative_time_ms =
            super::audio_processing::process_prelude(&tx, prelude_fn, cumulative_time_ms).await;
    }

    // Process each speaker line with real dia-voice TTS synthesis
    // cumulative_time_ms now properly starts after prelude
    for (chunk_index, line) in lines.into_iter().enumerate() {
        cumulative_time_ms = super::speaker_processing::process_speaker_line(
            &tx,
            &pool,
            line,
            voice_clone_path.as_deref(),
            chunk_index,
            cumulative_time_ms,
        )
        .await;

        if tx.is_closed() {
            break; // Receiver dropped, stop processing
        }
    }

    // Execute postlude function if provided
    if let Some(postlude_fn) = postlude {
        super::audio_processing::process_postlude(&tx, postlude_fn, cumulative_time_ms).await;
    }
}
