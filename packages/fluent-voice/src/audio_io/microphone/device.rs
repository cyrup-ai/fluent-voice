use anyhow::Result;
#[cfg(feature = "microphone")]
use cpal::traits::{DeviceTrait, HostTrait};
use fluent_voice_domain::transcription::TranscriptionSegmentImpl;
use fluent_voice_domain::VoiceError;

pub async fn handle_list_devices() -> Result<TranscriptionSegmentImpl> {
    let host = cpal::default_host();
    let _device = host
        .default_input_device()
        .ok_or_else(|| VoiceError::ProcessingError("Failed to get input devices".to_string()))?;
    for device in host
        .input_devices()
        .map_err(|e| anyhow::anyhow!("Failed to get input devices: {}", e))?
    {
        if let Ok(name) = device.name() {
            tracing::info!(device_name = %name, "Input Device");
        } else {
            tracing::info!("Input Device: (name unavailable)");
        }
        if let Ok(configs) = device.supported_input_configs() {
            for config in configs {
                tracing::info!(
                    channels = config.channels(),
                    min_sr = config.min_sample_rate().0,
                    max_sr = config.max_sample_rate().0,
                    buffer_size = ?config.buffer_size(),
                    sample_format = ?config.sample_format(),
                    "Supported input config"
                );
            }
        } else {
            tracing::info!("No supported input configs or error retrieving them");
        }
        tracing::info!("");
    }
    Err(anyhow::anyhow!("Device listing completed"))
}
