//! Test real audio device enumeration and Studio Display Microphone detection.

use fluent_voice::audio_device_manager::AudioDeviceManager;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎤 Audio Device Manager Test");
    println!("============================");

    // Create audio device manager
    let manager = AudioDeviceManager::new()?;
    println!("✅ Audio device manager created successfully");

    // Enumerate all available devices
    println!("\n🔍 Enumerating all audio devices:");
    println!("----------------------------------");

    let devices = manager.enumerate_devices()?;
    println!("Found {} audio device(s):", devices.len());

    for (index, device) in devices.iter().enumerate() {
        println!("  {}. {}", index + 1, device);
        if let Some(ref config) = device.default_input_config {
            println!(
                "     Input config: {} channels, {} Hz, {:?}",
                config.channels(),
                config.sample_rate().0,
                config.sample_format()
            );
        }
    }

    // Look specifically for Studio Display Microphone
    println!("\n🎯 Looking for Studio Display Microphone:");
    println!("------------------------------------------");

    match manager.find_studio_display_microphone()? {
        Some((device, info)) => {
            println!("✅ Found Studio Display Microphone!");
            println!("   Device: {}", info);

            // Validate the device
            match manager.validate_device(&device) {
                Ok(()) => println!("   ✅ Device validation successful"),
                Err(e) => println!("   ❌ Device validation failed: {}", e),
            }

            if let Some(ref config) = info.default_input_config {
                println!("   📊 Input capabilities:");
                println!("      - Channels: {}", config.channels());
                println!("      - Sample rate: {} Hz", config.sample_rate().0);
                println!("      - Sample format: {:?}", config.sample_format());
            }
        }
        None => {
            println!("❌ Studio Display Microphone not found");
            println!("   Available microphones:");
            for device in &devices {
                if device.supports_input {
                    println!("   - {}", device.name);
                }
            }
        }
    }

    // Get preferred microphone (Studio Display or fallback)
    println!("\n🎤 Getting preferred microphone:");
    println!("--------------------------------");

    match manager.get_preferred_microphone() {
        Ok((device, info)) => {
            println!("✅ Selected microphone: {}", info);

            if info.name == "Studio Display Microphone" {
                println!("   🎯 Using Studio Display Microphone (preferred)");
            } else {
                println!("   📱 Using fallback microphone: {}", info.name);
            }

            // Final validation
            match manager.validate_device(&device) {
                Ok(()) => println!("   ✅ Final device validation successful"),
                Err(e) => println!("   ❌ Final device validation failed: {}", e),
            }
        }
        Err(e) => {
            println!("❌ No microphone available: {}", e);
        }
    }

    println!("\n🎯 Audio Device Test Complete!");
    println!("===============================");
    println!("Real audio devices have been enumerated and tested.");
    println!("No mocked or simulated data was used in this test.");

    Ok(())
}
