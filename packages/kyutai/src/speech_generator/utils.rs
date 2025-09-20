//! Utility functions for speech generation

use super::error::SpeechGenerationError;
use std::path::Path;

/// Validate that a model file exists and is a valid safetensors file
pub fn validate_model_file<P: AsRef<Path>>(
    path: P,
    model_type: &str,
) -> Result<(), SpeechGenerationError> {
    let path = path.as_ref();

    // Check if file exists
    if !path.exists() {
        return Err(SpeechGenerationError::ModelLoading(format!(
            "{} file not found: {}",
            model_type,
            path.display()
        )));
    }

    // Check if file is readable
    if let Err(e) = std::fs::File::open(path) {
        return Err(SpeechGenerationError::ModelLoading(format!(
            "Cannot read {} file {}: {}",
            model_type,
            path.display(),
            e
        )));
    }

    // Validate safetensors format by attempting to read header
    match std::fs::read(path) {
        Ok(data) => {
            if data.len() < 8 {
                return Err(SpeechGenerationError::ModelLoading(format!(
                    "{} file {} is too small to be a valid safetensors file",
                    model_type,
                    path.display()
                )));
            }

            // Basic safetensors validation - check for valid header length
            let header_len = u64::from_le_bytes([
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            ]);

            if header_len as usize + 8 > data.len() {
                return Err(SpeechGenerationError::ModelLoading(format!(
                    "{} file {} has invalid safetensors header",
                    model_type,
                    path.display()
                )));
            }
        }
        Err(e) => {
            return Err(SpeechGenerationError::ModelLoading(format!(
                "Failed to read {} file {}: {}",
                model_type,
                path.display(),
                e
            )));
        }
    }

    Ok(())
}
