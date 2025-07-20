//! Test script to create a minimal valid wakeword model file

use koffee::{
    ModelWeights,
    wakewords::{
        WakewordSave,
        wakeword_model::{ModelType, TensorData, WakewordModel},
    },
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a minimal valid model
    let model = WakewordModel {
        version: 1, // Must match MODEL_VERSION in wakeword_model.rs
        labels: vec!["none".to_string(), "wake".to_string()],
        train_size: 1,
        kfc_size: (16, 1), // 16 coefficients, 1 frame
        m_type: ModelType::Tiny,
        weights: ModelWeights::from(HashMap::from([
            ("dense/kernel".to_string(), TensorData::F32(vec![0.0; 16])),
            ("dense/bias".to_string(), TensorData::F32(vec![0.0])),
            ("output/kernel".to_string(), TensorData::F32(vec![0.0; 2])),
            ("output/bias".to_string(), TensorData::F32(vec![0.0; 2])),
        ])),
        rms_level: 0.1,
    };

    // Save it to a file
    let path = "minimal_model.rpw";
    model.save_to_file(path)?;
    println!("✅ Created minimal model at: {}", path);

    Ok(())
}
