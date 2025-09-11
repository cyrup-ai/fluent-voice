//! Train "Syrup" Wake Word Model
//!
//! This example trains a wake word model for "syrup" using the existing
//! training samples in the syrup_training/ directory.
//!
//! Usage:
//!   cargo run --example train_syrup -- [OPTIONS]
//!
//! Options:
//!   -m, --model-type TYPE  Model type: 0=Tiny, 1=Small (default), 2=Medium, 3=Large
//!   -l, --learning-rate LR Learning rate (default: 0.001)
//!   -e, --epochs N         Number of training epochs (default: 20)
//!   -b, --batch-size N     Batch size (default: 2)
//!   -o, --output FILE      Output model file (default: syrup_new.rpw)
//!   -h, --help             Print help

use clap::Parser;
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

/// Command line arguments for the training script
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model type: 0=Tiny, 1=Small (default), 2=Medium, 3=Large
    #[arg(short, long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(0..=3))]
    model_type: u8,

    /// Learning rate
    #[arg(short, long, default_value_t = 0.001)]
    learning_rate: f64,

    /// Number of training epochs
    #[arg(short, long, default_value_t = 20)]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value_t = 2)]
    batch_size: usize,

    /// Output model file
    #[arg(short, long, default_value = "syrup_new.rpw")]
    output: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍯 Training 'Syrup' Wake Word Model");
    println!("==================================");
    println!();

    // Parse command line arguments
    let args = Args::parse();

    // Training configuration
    let training_dir = "syrup_training";
    let output_model = &args.output;
    let model_type = match args.model_type {
        0 => ModelType::Tiny,
        1 => ModelType::Small,
        2 => ModelType::Medium,
        3 => ModelType::Large,
        _ => ModelType::Small, // Default to Small
    };

    // Check if training directory exists
    if !Path::new(training_dir).exists() {
        return Err(format!("Training directory '{}' not found!", training_dir).into());
    }

    // Load training data
    println!("📂 Loading training data from: {}", training_dir);
    let train_data = load_training_data(training_dir)?;

    println!("✅ Loaded {} audio files", train_data.len());

    // Count positive and negative samples
    let positive_count = train_data.keys().filter(|k| k.contains("[syrup]")).count();
    let negative_count = train_data.keys().filter(|k| k.contains("noise")).count();

    println!("   - Positive samples (syrup): {}", positive_count);
    println!("   - Negative samples (noise): {}", negative_count);
    println!();

    if positive_count < 5 {
        eprintln!(
            "⚠️  Warning: Only {} positive samples found. Recommend at least 10 for good accuracy.",
            positive_count
        );
    }

    // Configure training with optimal settings for wake word detection
    let train_options = WakewordModelTrainOptions {
        model_type: model_type as u8,
        lr: args.learning_rate,
        epochs: args.epochs,
        batch_size: args.batch_size,
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
    println!("   This may take 2-5 minutes...");

    let start_time = std::time::Instant::now();

    // Use same data for validation (suitable for small dataset)
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
    let output_path = Path::new(output_model);
    let output_dir = output_path.parent().unwrap_or_else(|| Path::new("."));
    let output_filename = output_path
        .file_name()
        .unwrap_or_else(|| "syrup.rpw".as_ref());

    // Create output directory if it doesn't exist
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }

    let output_path = output_dir.join(output_filename);
    let output_path_str = output_path.to_str().unwrap_or("unknown");

    println!("💾 Saving model to: {}", output_path.display());
    model.save_to_file(&output_path_str)?;
    println!("✅ Model saved successfully to: {}", output_path.display());

    // Create a symlink to the default location if not already there
    let default_path = Path::new("syrup.rpw");
    if output_path != default_path {
        if default_path.exists() {
            let backup_path = Path::new("syrup_old.rpw");
            println!("🔄 Backing up old model to: {}", backup_path.display());
            if backup_path.exists() {
                fs::remove_file(backup_path)?;
            }
            fs::rename(default_path, backup_path)?;
        }

        // Create a symlink if possible, otherwise copy the file
        #[cfg(unix)]
        std::os::unix::fs::symlink(&output_path, default_path).unwrap_or_else(|_| {
            println!("⚠️  Could not create symlink, copying file instead");
            fs::copy(&output_path, default_path).expect("Failed to copy model file");
        });

        #[cfg(not(unix))]
        {
            println!(
                "⚠️  Copying model to default location: {}",
                default_path.display()
            );
            fs::copy(&output_path, default_path).expect("Failed to copy model file");
        }

        println!(
            "🔗 Created shortcut: {} -> {}",
            default_path.display(),
            output_path.display()
        );
    }

    println!("📋 Moving new model to syrup.rpw");
    fs::rename(output_model, "syrup.rpw")?;

    println!();
    println!("🎯 Success! The syrup wake word model is ready.");
    println!("📝 Next steps:");
    println!("   1. Test the model: cargo run --example cyrup_wake");
    println!("   2. Use in voice applications for 'syrup' wake word detection");

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
