//! Utility functions for model operations

use crate::tokenizer::KyutaiTokenizer;
use candle_core::{Device, Result};
use std::path::Path;

/// Load model weights from safetensors file
pub fn load_model_weights(model_path: &str, _device: &Device) -> Result<candle_nn::VarMap> {
    let mut var_map = candle_nn::VarMap::new();
    var_map.load(model_path)?;
    Ok(var_map)
}

/// Save model weights to safetensors file
pub fn save_model_weights(var_map: &candle_nn::VarMap, model_path: &str) -> Result<()> {
    var_map.save(model_path)?;
    Ok(())
}

/// Create a production tokenizer from file
pub fn create_tokenizer<P: AsRef<Path>>(tokenizer_path: P) -> crate::Result<KyutaiTokenizer> {
    KyutaiTokenizer::from_file(tokenizer_path)
}

/// Create tokenizer from pretrained model
#[cfg(feature = "http")]
pub fn create_pretrained_tokenizer(model_name: &str) -> crate::Result<KyutaiTokenizer> {
    KyutaiTokenizer::from_pretrained(model_name)
}
