//! Training functionality for wake-word models

use std::{collections::HashMap, fs};

use crate::{
    ModelType, Result,
    wakewords::{
        WakewordSave,
        nn::{WakewordModelTrain, WakewordModelTrainOptions},
    },
};

/// Train a wake-word model from a directory of wav files
pub fn train_dir(input_dir: &str, output_path: &str, model_type: ModelType) -> Result<()> {
    // Read wav files from directory
    let train_data = load_wav_files(input_dir)?;
    let val_data = load_wav_files(input_dir)?; // TODO: Split train/val properly

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

    println!("Model trained and saved to: {output_path}");
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
