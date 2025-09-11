//! List available audio input devices
//!
//! This example lists all available audio input devices on the system.

use cpal::traits::{DeviceTrait, HostTrait};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎙️ Available audio input devices:");
    println!("==================================");

    let host = cpal::default_host();

    // List all available input devices
    let devices = host.input_devices()?;

    for (i, device) in devices.enumerate() {
        let name = device.name()?;
        let default = match host.default_input_device() {
            Some(default_device) => {
                if device.name().unwrap_or_default() == default_device.name().unwrap_or_default() {
                    " (default)"
                } else {
                    ""
                }
            }
            None => "",
        };

        println!("{}. {}{}", i + 1, name, default);

        // List supported input configurations
        if let Ok(configs) = device.supported_input_configs() {
            println!("   Supported configurations:");
            for (j, config) in configs.enumerate() {
                println!(
                    "   {}.{: <3} Sample rate: {}-{} Hz, Channels: {}, Format: {:?}",
                    i + 1,
                    j + 1,
                    config.min_sample_rate().0,
                    config.max_sample_rate().0,
                    config.channels(),
                    config.sample_format()
                );
            }
        } else {
            println!("   No supported input configurations found");
        }

        println!();
    }

    Ok(())
}
