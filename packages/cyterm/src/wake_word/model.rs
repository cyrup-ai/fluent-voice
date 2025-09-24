//! Wake word model implementation.

use super::error::{Result, WakeWordError};
use crate::features::FEATURE_DIM;
use std::path::Path;

/// Wake word detection model.
#[derive(Debug, Clone)]
pub struct KwModel {
    /// Model weights as a flat vector.
    /// Size should match FEATURE_DIM for dot product computation.
    weights: Vec<f32>,

    /// Model bias term.
    bias: f32,
}

impl KwModel {
    /// Create a new wake word model with given weights and bias.
    pub fn new(weights: Vec<f32>, bias: f32) -> Result<Self> {
        if weights.len() != FEATURE_DIM {
            return Err(WakeWordError::ModelLoadFailed {
                reason: format!(
                    "Weight vector size mismatch: expected {}, got {}",
                    FEATURE_DIM,
                    weights.len()
                ),
            });
        }

        Ok(Self { weights, bias })
    }

    /// Load a wake word model from a file.
    ///
    /// The file format is expected to be:
    /// - First 4 bytes: bias as f32 (little-endian)
    /// - Remaining bytes: weights as f32 values (little-endian)
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = std::fs::read(path.as_ref()).map_err(|e| WakeWordError::ModelLoadFailed {
            reason: format!("Failed to read model file: {}", e),
        })?;

        if data.len() < 4 {
            return Err(WakeWordError::ModelLoadFailed {
                reason: "Model file too small".to_string(),
            });
        }

        // Read bias (first 4 bytes)
        let bias = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);

        // Read weights (remaining bytes)
        let weight_bytes = &data[4..];
        if weight_bytes.len() % 4 != 0 {
            return Err(WakeWordError::ModelLoadFailed {
                reason: "Invalid model file: weight data not aligned to f32".to_string(),
            });
        }

        let expected_weight_count = FEATURE_DIM;
        let actual_weight_count = weight_bytes.len() / 4;

        if actual_weight_count != expected_weight_count {
            return Err(WakeWordError::ModelLoadFailed {
                reason: format!(
                    "Weight count mismatch: expected {}, got {}",
                    expected_weight_count, actual_weight_count
                ),
            });
        }

        let mut weights = Vec::with_capacity(actual_weight_count);
        for chunk in weight_bytes.chunks_exact(4) {
            let weight = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            weights.push(weight);
        }

        Ok(Self { weights, bias })
    }

    /// Load a wake word model from embedded bytes.
    pub fn load_from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(WakeWordError::ModelLoadFailed {
                reason: "Model data too small".to_string(),
            });
        }

        // Read bias (first 4 bytes)
        let bias = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);

        // Read weights (remaining bytes)
        let weight_bytes = &data[4..];
        if weight_bytes.len() % 4 != 0 {
            return Err(WakeWordError::ModelLoadFailed {
                reason: "Invalid model data: weight data not aligned to f32".to_string(),
            });
        }

        let expected_weight_count = FEATURE_DIM;
        let actual_weight_count = weight_bytes.len() / 4;

        if actual_weight_count != expected_weight_count {
            return Err(WakeWordError::ModelLoadFailed {
                reason: format!(
                    "Weight count mismatch: expected {}, got {}",
                    expected_weight_count, actual_weight_count
                ),
            });
        }

        let mut weights = Vec::with_capacity(actual_weight_count);
        for chunk in weight_bytes.chunks_exact(4) {
            let weight = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            weights.push(weight);
        }

        Ok(Self { weights, bias })
    }

    /// Compute dot product with feature vector and return raw score.
    pub fn dot(&self, features: &[f32; FEATURE_DIM]) -> ModelOutput {
        let mut sum = self.bias;
        for (weight, &feature) in self.weights.iter().zip(features.iter()) {
            sum += weight * feature;
        }
        ModelOutput { raw_score: sum }
    }

    /// Get the model weights.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Get the model bias.
    pub fn bias(&self) -> f32 {
        self.bias
    }
}

/// Output from the wake word model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelOutput {
    raw_score: f32,
}

impl ModelOutput {
    /// Apply sigmoid activation to get probability score (0.0 to 1.0).
    pub fn sigmoid(self) -> f32 {
        1.0 / (1.0 + (-self.raw_score).exp())
    }

    /// Get the raw score before activation.
    pub fn raw_score(self) -> f32 {
        self.raw_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let weights = vec![0.5; FEATURE_DIM];
        let bias = 0.1;
        let model = KwModel::new(weights.clone(), bias).expect("Failed to create model");

        assert_eq!(model.weights().len(), FEATURE_DIM);
        assert_eq!(model.bias(), bias);
    }

    #[test]
    fn test_dot_product() {
        let weights = vec![1.0; FEATURE_DIM];
        let bias = 0.0;
        let model = KwModel::new(weights, bias).expect("Failed to create model");

        let features = [0.5; FEATURE_DIM];
        let output = model.dot(&features);
        let expected = FEATURE_DIM as f32 * 0.5;
        assert!((output.raw_score() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        let output = ModelOutput { raw_score: 0.0 };
        assert!((output.sigmoid() - 0.5).abs() < 1e-6);

        let output = ModelOutput {
            raw_score: f32::INFINITY,
        };
        assert!((output.sigmoid() - 1.0).abs() < 1e-6);

        let output = ModelOutput {
            raw_score: f32::NEG_INFINITY,
        };
        assert!((output.sigmoid() - 0.0).abs() < 1e-6);
    }
}
