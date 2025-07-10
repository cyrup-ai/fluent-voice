//! Train "Syrup" Wake Word Model
//!
//! This example trains a wake word model for "syrup" using the existing
//! training samples in the syrup_training/ directory.
//!
//! Usage: cargo run --example train_syrup

use koffee::{
    ModelType,
    wakewords::{WakewordSave, nn::{WakewordModelTrainOptions, WakewordModelTrain}},
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍯 Training 'Syrup' Wake Word Model");
    println!("==================================");
    println!();

    // Training configuration
    let training_dir = "syrup_training";
    let output_model = "syrup_new.rpw";
    let model_type = ModelType::Small; // Small model for quick training and response

    // Check if training directory exists
    if !Path::new(training_dir).exists() {
        return Err(format!("Training directory '{}' not found!", training_dir).into());
    }

    // Load training data
    println!("📂 Loading training data from: {}", training_dir);
    let train_data = load_training_data(training_dir)?;
    
    println!("✅ Loaded {} audio files", train_data.len());
    
    // Count positive and negative samples
    let positive_count = train_data.keys()
        .filter(|k| k.contains("[syrup]"))
        .count();
    let negative_count = train_data.keys()
        .filter(|k| k.contains("noise"))
        .count();
    
    println!("   - Positive samples (syrup): {}", positive_count);
    println!("   - Negative samples (noise): {}", negative_count);
    println!();

    if positive_count < 5 {
        eprintln!("⚠️  Warning: Only {} positive samples found. Recommend at least 10 for good accuracy.", positive_count);
    }

    // Configure training with optimal settings for wake word detection
    let train_options = WakewordModelTrainOptions {
        model_type: model_type as u8,
        lr: 0.001,
        epochs: 20, // Reduced epochs for stability
        batch_size: 2, // Even smaller batch size for very limited data
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
    ).map_err(|e| format!("Training failed: {:?}", e))?;
    
    let elapsed = start_time.elapsed();
    println!("✅ Training completed in {:.1} seconds", elapsed.as_secs_f32());
    println!();

    // Save the model
    println!("💾 Saving model to: {}", output_model);
    model.save_to_file(output_model)?;
    
    println!("✅ Model saved successfully!");
    
    // Replace the old model
    if Path::new("syrup.rpw").exists() {
        println!("🔄 Backing up old model to syrup_old.rpw");
        fs::rename("syrup.rpw", "syrup_old.rpw")?;
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
fn load_training_data(dir_path: &str) -> Result<HashMap<String, Vec<u8>>, Box<dyn std::error::Error>> {
    let mut data = HashMap::new();
    
    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("wav") {
            let filename = path.file_name()
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