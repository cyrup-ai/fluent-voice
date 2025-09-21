//! Training functionality for wake-word models

use rand::seq::SliceRandom;
use std::{collections::HashMap, fs, path::Path};

use crate::{
    ModelType, Result,
    wakewords::{
        WakewordSave,
        nn::{WakewordModelTrain, WakewordModelTrainOptions},
    },
};

/// Train a wake-word model from a directory of wav files
pub fn train_dir(input_dir: &Path, output_path: &Path, model_type: ModelType) -> Result<()> {
    // Read wav files from directory
    let all_data = load_wav_files_from_path(input_dir)?;

    // Implement proper stratified train/validation split (80/20)
    let (train_data, val_data) = split_training_data(all_data, 0.8)?;

    // Configure training options
    let model_type_u8 = match model_type {
        ModelType::Tiny => 0,
        ModelType::Small => 1,
        ModelType::Medium => 2,
        ModelType::Large => 3,
    };
    let opts = WakewordModelTrainOptions {
        model_type: model_type_u8,
        ..Default::default()
    };

    // Train the model
    let model = WakewordModelTrain(train_data, val_data, None, opts)
        .map_err(|e| format!("Training failed: {e:?}"))?;

    // Save the trained model
    model
        .save_to_file(output_path)
        .map_err(|e| format!("Failed to save model: {e}"))?;

    println!("Model trained and saved to: {}", output_path.display());
    Ok(())
}

/// Load wav files from directory into training data format
fn load_wav_files(dir_path: &str) -> Result<HashMap<String, Vec<u8>>> {
    let mut data = HashMap::new();

    let dir =
        fs::read_dir(dir_path).map_err(|e| format!("Failed to read directory {dir_path}: {e}"))?;

    for entry in dir {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {e}"))?;
        let path = entry.path();

        if let Some(ext) = path.extension()
            && ext == "wav"
        {
            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| "Invalid filename".to_string())?;

            let wav_data = fs::read(&path)
                .map_err(|e| format!("Failed to read wav file {}: {e}", path.display()))?;

            data.insert(filename.to_string(), wav_data);
        }
    }

    if data.is_empty() {
        return Err(format!("No wav files found in directory: {dir_path}"));
    }

    Ok(data)
}

/// Load wav files from a Path directory
fn load_wav_files_from_path(dir_path: &Path) -> Result<HashMap<String, Vec<u8>>> {
    let dir_path_str = dir_path
        .to_str()
        .ok_or_else(|| format!("Directory path contains invalid UTF-8: {:?}", dir_path))?;
    load_wav_files(dir_path_str)
}

/// Split training data into train and validation sets with stratified sampling
fn split_training_data(
    data: HashMap<String, Vec<u8>>,
    train_ratio: f32,
) -> Result<(HashMap<String, Vec<u8>>, HashMap<String, Vec<u8>>)> {
    let mut positive_samples = Vec::new();
    let mut negative_samples = Vec::new();

    // Separate positive and negative samples based on filename patterns
    for (filename, audio_data) in data {
        if filename.contains("noise")
            || filename.contains("negative")
            || filename.contains("background")
        {
            negative_samples.push((filename, audio_data));
        } else {
            positive_samples.push((filename, audio_data));
        }
    }

    // Shuffle samples for random distribution
    let mut rng = rand::rng();
    positive_samples.shuffle(&mut rng);
    negative_samples.shuffle(&mut rng);

    // Calculate split indices
    let pos_train_count = (positive_samples.len() as f32 * train_ratio) as usize;
    let neg_train_count = (negative_samples.len() as f32 * train_ratio) as usize;

    // Split positive samples
    let (train_pos, val_pos) = positive_samples.split_at(pos_train_count);

    // Split negative samples
    let (train_neg, val_neg) = negative_samples.split_at(neg_train_count);

    // Combine train and validation sets
    let mut train_data = HashMap::new();
    let mut val_data = HashMap::new();

    for (filename, audio_data) in train_pos.iter().chain(train_neg.iter()) {
        train_data.insert(filename.clone(), audio_data.clone());
    }

    for (filename, audio_data) in val_pos.iter().chain(val_neg.iter()) {
        val_data.insert(filename.clone(), audio_data.clone());
    }

    // Ensure we have data in both sets
    if train_data.is_empty() {
        return Err("No training data available after split".to_string());
    }

    if val_data.is_empty() {
        return Err("No validation data available after split".to_string());
    }

    Ok((train_data, val_data))
}
