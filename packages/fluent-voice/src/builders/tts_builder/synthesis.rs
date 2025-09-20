//! Internal TTS synthesis functions

use dia::voice::VoicePool;
use fluent_voice_domain::VoiceError;
use std::sync::Arc;

/// Internal function for real TTS speech synthesis using dia-voice
/// This function is separate from the impl block to avoid generic parameter issues
pub(super) async fn synthesize_speech_internal(
    pool: &Arc<VoicePool>,
    speaker_id: &str,
    text: &str,
    voice_clone_path: Option<&std::path::Path>,
) -> Result<Vec<u8>, VoiceError> {
    // Use dia-voice engine for real TTS synthesis
    use dia::voice::{Conversation, DiaSpeaker, VoiceClone};

    // Create basic voice data for the speaker
    use candle_core::{Device, Tensor};
    use dia::voice::codec::VoiceData;
    use std::path::PathBuf;

    // Load voice data from clone path if available, otherwise use default
    let voice_data = if let Some(clone_path) = voice_clone_path {
        // Load real voice data from the provided path
        match pool.load_voice(speaker_id, clone_path) {
            Ok(loaded_data) => loaded_data,
            Err(e) => {
                return Err(VoiceError::Configuration(format!(
                    "Failed to load voice clone from path {:?}: {}",
                    clone_path, e
                )));
            }
        }
    } else {
        // Fallback to default voice data creation
        let device = Device::Cpu;
        let codes = Tensor::zeros((1, 1), candle_core::DType::F32, &device)
            .map_err(|e| VoiceError::Configuration(format!("Failed to create tensor: {}", e)))?;
        Arc::new(VoiceData {
            codes,
            sample_rate: 24000,
            source_path: PathBuf::from("default_voice.wav"),
        })
    };

    // Create voice clone and speaker
    let voice_clone = VoiceClone::new(speaker_id.to_string(), voice_data);
    let speaker = DiaSpeaker { voice_clone };

    // Create conversation and generate TTS audio using dia-voice engine
    let conversation = Conversation::new(text.to_string(), speaker, pool.clone())
        .await
        .map_err(|e| VoiceError::Synthesis(format!("Failed to create TTS conversation: {}", e)))?;

    // Generate speech audio using dia-voice engine
    let voice_player = conversation
        .internal_generate()
        .await
        .map_err(|e| VoiceError::Synthesis(format!("Failed to generate speech: {}", e)))?;

    // Extract audio bytes from dia-voice engine
    let audio_bytes = voice_player
        .to_bytes()
        .await
        .map_err(|e| VoiceError::Synthesis(format!("Failed to extract speech bytes: {}", e)))?;

    // Log the synthesis for debugging
    tracing::info!(size_bytes = audio_bytes.len(), speaker_id = %speaker_id, text = %text, "Synthesized audio");

    Ok(audio_bytes)
}
