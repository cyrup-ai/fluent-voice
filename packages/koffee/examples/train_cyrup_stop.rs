//! Train "Cyrup Stop" Wake Word Model
//!
//! This example shows how to train a custom wake word model for "cyrup stop"
//! to be used as an unwake command.
//!
//! Usage:
//!   1. First, record audio samples saying "cyrup stop"
//!   2. Place them in cyrup_stop_training/ directory with format:
//!      - cyrup_stop_00[cyrup stop].wav
//!      - cyrup_stop_01[cyrup stop].wav
//!      - etc...
//!   3. Add some noise samples:
//!      - noise0.wav
//!      - noise1.wav
//!   4. Run: cargo run --example train_cyrup_stop

use koffee::{
    ModelType,
    wakewords::{
        WakewordSave,
        nn::{WakewordModelTrain, WakewordModelTrainOptions},
    },
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Training 'Cyrup Stop' Wake Word Model");
    println!("=======================================");
    println!();

    // Training configuration
    let training_dir = "cyrup_stop_training";
    let output_model = "cyrup_stop.rpw";
    let model_type = ModelType::Small; // Small model for quick training

    // Check if training directory exists
    if !Path::new(training_dir).exists() {
        eprintln!("❌ Training directory '{}' not found!", training_dir);
        eprintln!();
        eprintln!("📝 To create training data:");
        eprintln!("   1. Create directory: mkdir {}", training_dir);
        eprintln!("   2. Record 10-15 samples of 'cyrup stop' (1-2 seconds each)");
        eprintln!(
            "   3. Name them: cyrup_stop_00[cyrup stop].wav, cyrup_stop_01[cyrup stop].wav, etc."
        );
        eprintln!("   4. Add 2-3 noise samples: noise0.wav, noise1.wav, etc.");
        eprintln!();
        eprintln!("💡 Example using SoX to record:");
        eprintln!("   sox -d -r 16000 -c 1 -b 16 cyrup_stop_00[cyrup stop].wav trim 0 2");

        return Ok(());
    }

    // Load training data
    println!("📂 Loading training data from: {}", training_dir);
    let train_data = load_training_data(training_dir)?;

    println!("✅ Loaded {} audio files", train_data.len());

    // Count positive and negative samples
    let positive_count = train_data
        .keys()
        .filter(|k| k.contains("[cyrup stop]"))
        .count();
    let negative_count = train_data.keys().filter(|k| k.contains("noise")).count();

    println!("   - Positive samples (cyrup stop): {}", positive_count);
    println!("   - Negative samples (noise): {}", negative_count);
    println!();

    if positive_count < 5 {
        eprintln!(
            "⚠️  Warning: Only {} positive samples found. Recommend at least 10 for good accuracy.",
            positive_count
        );
    }

    // Configure training
    let train_options = WakewordModelTrainOptions {
        model_type: model_type as u8,
        lr: 0.001,
        epochs: 20, // More epochs for better accuracy
        batch_size: 8,
        ..Default::default()
    };

    println!("🔧 Training configuration:");
    println!("   - Model type: {:?}", model_type);
    println!("   - Learning rate: {}", train_options.lr);
    println!("   - Epochs: {}", train_options.epochs);
    println!("   - Batch size: {}", train_options.batch_size);
    println!();

    // Train the model
    println!("🚀 Starting training...");
    println!("   This may take a few minutes...");

    let start_time = std::time::Instant::now();

    // For now, use the same data for validation (in production, split properly)
    let val_data = train_data.clone();

    let model = WakewordModelTrain(
        train_data,
        val_data,
        None, // No pre-trained model
        train_options,
    )
    .map_err(|e| format!("Training failed: {:?}", e))?;

    let elapsed = start_time.elapsed();
    println!(
        "✅ Training completed in {:.1} seconds",
        elapsed.as_secs_f32()
    );
    println!();

    // Save the model
    println!("💾 Saving model to: {}", output_model);
    model.save_to_file(output_model)?;

    println!("✅ Model saved successfully!");
    println!();
    println!("📝 Next steps:");
    println!("   1. Test the model: cargo run --example test_cyrup_models");
    println!("   2. Use in your app with both 'syrup.rpw' and 'cyrup_stop.rpw'");

    Ok(())
}

/// Load WAV files from directory
fn load_training_data(
    dir_path: &str,
) -> Result<HashMap<String, Vec<u8>>, Box<dyn std::error::Error>> {
    let mut data = HashMap::new();

    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("wav") {
            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or("Invalid filename")?;

            println!("   Loading: {}", filename);

            let wav_data = fs::read(&path)?;
            data.insert(filename.to_string(), wav_data);
        }
    }

    if data.is_empty() {
        return Err(format!("No WAV files found in {}", dir_path).into());
    }

    Ok(data)
}
