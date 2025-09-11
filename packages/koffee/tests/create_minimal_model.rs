//! Test script to create a minimal valid wakeword model file

use indexmap::IndexMap;
use koffee::{
    ModelWeights,
    wakewords::{
        WakewordSave,
        wakeword_model::{ModelType, TensorData, WakewordModel},
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a minimal valid model
    let model = WakewordModel {
        version: 1, // Must match MODEL_VERSION in wakeword_model.rs
        labels: vec!["none".to_string(), "wake".to_string()],
        train_size: 1,
        kfc_size: (16, 1), // 16 coefficients, 1 frame
        m_type: ModelType::Tiny,
        weights: ModelWeights::Map(IndexMap::from([
            (
                "dense/kernel".to_string(),
                TensorData {
                    bytes: vec![0u8; 16 * 4], // 16 f32 values = 64 bytes
                    dims: vec![16],
                    d_type: "f32".to_string(),
                },
            ),
            (
                "dense/bias".to_string(),
                TensorData {
                    bytes: vec![0u8; 4], // 1 f32 value = 4 bytes
                    dims: vec![1],
                    d_type: "f32".to_string(),
                },
            ),
            (
                "output/kernel".to_string(),
                TensorData {
                    bytes: vec![0u8; 2 * 4], // 2 f32 values = 8 bytes
                    dims: vec![2],
                    d_type: "f32".to_string(),
                },
            ),
            (
                "output/bias".to_string(),
                TensorData {
                    bytes: vec![0u8; 2 * 4], // 2 f32 values = 8 bytes
                    dims: vec![2],
                    d_type: "f32".to_string(),
                },
            ),
        ])),
        rms_level: 0.1,
    };

    // Save it to a file
    let path = "minimal_model.rpw";
    model.save_to_file(path)?;
    println!("✅ Created minimal model at: {}", path);

    Ok(())
}
