//! Real audio device management for microphone enumeration and selection.

#[cfg(feature = "microphone")]
use cpal::{
    Device, Host, SupportedStreamConfig,
    traits::{DeviceTrait, HostTrait},
};
use fluent_voice_domain::VoiceError;
use std::fmt;

/// Audio device information.
#[derive(Debug, Clone)]
pub struct AudioDeviceInfo {
    /// Device name as reported by the system.
    pub name: String,
    /// Whether this device supports input (microphone).
    pub supports_input: bool,
    /// Whether this device supports output (speakers).
    pub supports_output: bool,
    /// Default input configuration if supported.
    pub default_input_config: Option<SupportedStreamConfig>,
}

impl fmt::Display for AudioDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (Input: {}, Output: {})",
            self.name, self.supports_input, self.supports_output
        )
    }
}

/// Audio device manager for real microphone enumeration and selection.
pub struct AudioDeviceManager {
    host: Host,
}

impl AudioDeviceManager {
    /// Create a new audio device manager.
    pub fn new() -> Result<Self, VoiceError> {
        let host = cpal::default_host();
        Ok(Self { host })
    }

    /// Enumerate all available audio devices.
    pub fn enumerate_devices(&self) -> Result<Vec<AudioDeviceInfo>, VoiceError> {
        let mut devices = Vec::new();

        let device_iter = self.host.devices().map_err(|e| {
            VoiceError::Configuration(format!("Failed to enumerate audio devices: {}", e))
        })?;

        for device in device_iter {
            let name = device.name().map_err(|e| {
                VoiceError::Configuration(format!("Failed to get device name: {}", e))
            })?;

            // Check input capabilities
            let supports_input = device.default_input_config().is_ok();
            let supports_output = device.default_output_config().is_ok();

            let default_input_config = if supports_input {
                device.default_input_config().ok()
            } else {
                None
            };

            devices.push(AudioDeviceInfo {
                name,
                supports_input,
                supports_output,
                default_input_config,
            });
        }

        Ok(devices)
    }

    /// Find the Studio Display Microphone device specifically.
    pub fn find_studio_display_microphone(
        &self,
    ) -> Result<Option<(Device, AudioDeviceInfo)>, VoiceError> {
        let device_iter = self.host.devices().map_err(|e| {
            VoiceError::Configuration(format!("Failed to enumerate audio devices: {}", e))
        })?;

        for device in device_iter {
            let name = device.name().map_err(|e| {
                VoiceError::Configuration(format!("Failed to get device name: {}", e))
            })?;

            // Check for exact match with Studio Display Microphone
            if name == "Studio Display Microphone" {
                // Verify it supports input
                let supports_input = device.default_input_config().is_ok();
                if !supports_input {
                    continue;
                }

                let default_input_config = device.default_input_config().ok();

                let device_info = AudioDeviceInfo {
                    name: name.clone(),
                    supports_input: true,
                    supports_output: device.default_output_config().is_ok(),
                    default_input_config,
                };

                return Ok(Some((device, device_info)));
            }
        }

        Ok(None)
    }

    /// Find any microphone device as fallback.
    pub fn find_default_microphone(&self) -> Result<Option<(Device, AudioDeviceInfo)>, VoiceError> {
        // First try the system default input device
        if let Some(device) = self.host.default_input_device() {
            let name = device.name().map_err(|e| {
                VoiceError::Configuration(format!("Failed to get device name: {}", e))
            })?;

            let default_input_config = device.default_input_config().ok();

            let device_info = AudioDeviceInfo {
                name: name.clone(),
                supports_input: true,
                supports_output: device.default_output_config().is_ok(),
                default_input_config,
            };

            return Ok(Some((device, device_info)));
        }

        // If no default device, find any input device
        let device_iter = self.host.devices().map_err(|e| {
            VoiceError::Configuration(format!("Failed to enumerate audio devices: {}", e))
        })?;

        for device in device_iter {
            if device.default_input_config().is_ok() {
                let name = device.name().map_err(|e| {
                    VoiceError::Configuration(format!("Failed to get device name: {}", e))
                })?;

                let default_input_config = device.default_input_config().ok();

                let device_info = AudioDeviceInfo {
                    name: name.clone(),
                    supports_input: true,
                    supports_output: device.default_output_config().is_ok(),
                    default_input_config,
                };

                return Ok(Some((device, device_info)));
            }
        }

        Ok(None)
    }

    /// Get the best available microphone device with Studio Display preference.
    pub fn get_preferred_microphone(&self) -> Result<(Device, AudioDeviceInfo), VoiceError> {
        // First try Studio Display Microphone
        if let Some((device, info)) = self.find_studio_display_microphone()? {
            return Ok((device, info));
        }

        // Fall back to default microphone
        if let Some((device, info)) = self.find_default_microphone()? {
            return Ok((device, info));
        }

        Err(VoiceError::Configuration(
            "No microphone devices found on this system".to_string(),
        ))
    }

    /// Validate audio device connection and configuration.
    pub fn validate_device(&self, device: &Device) -> Result<(), VoiceError> {
        // Check if device is still available
        let name = device
            .name()
            .map_err(|e| VoiceError::Configuration(format!("Device no longer available: {}", e)))?;

        // Check if input is still supported
        device.default_input_config().map_err(|e| {
            VoiceError::Configuration(format!(
                "Device '{}' no longer supports audio input: {}",
                name, e
            ))
        })?;

        Ok(())
    }
}

impl Default for AudioDeviceManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|e| {
            tracing::error!("Failed to create audio device manager: {}", e);
            // Return a minimal fallback instance
            AudioDeviceManager {
                host: cpal::default_host(),
            }
        })
    }
}
