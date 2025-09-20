// src/utils.rs

use candle_core::{D, DType, Tensor, Result as CResult};
use std::collections::HashSet;

/// Adds sinusoidal embeddings to the input tensor.
///
/// This function adds sinusoidal position embeddings to the input tensor.
///
/// # Arguments
///
/// * `xs` - The input tensor.
///
/// # Returns
///
/// * `Result<Tensor>` - The tensor with added sinusoidal embeddings.
pub fn add_sin_embeddings(xs: &Tensor) -> CResult<Tensor> {
    let (b, t, d) = xs.dims3()?;
    let half_d = d / 2;
    let positions = Tensor::arange(0u32, t as u32, xs.device())?
        .to_dtype(DType::F32)?
        .unsqueeze(1)?;
    let freqs = Tensor::arange(0u32, half_d as u32, xs.device())?
        .to_dtype(DType::F32)?
        .unsqueeze(0)?;
    let half_d_tensor = Tensor::full(half_d as f32, freqs.shape(), freqs.device())?;
    let ten_thousand = Tensor::full(10000.0f32, freqs.shape(), freqs.device())?;
    let inv_freq = freqs
        .broadcast_div(&half_d_tensor)?
        .neg()?
        .broadcast_mul(&ten_thousand)?;
    let emb = positions.broadcast_mul(&inv_freq)?;
    let sin = emb.sin()?;
    let cos = emb.cos()?;
    let emb = Tensor::cat(&[sin, cos], D::Minus1)?
        .unsqueeze(0)?
        .broadcast_as((b, t, d))?;
    xs.broadcast_add(&emb)
}

/// Normalizes the input tensor along the last dimension.
///
/// This function computes the RMS normalization of the input tensor.
///
/// # Arguments
///
/// * `xs` - The input tensor.
/// * `alpha` - The scaling tensor.
/// * `eps` - The epsilon value for numerical stability.
///
/// # Returns
///
/// * `Result<Tensor>` - The normalized tensor.
pub fn rms_norm(xs: &Tensor, alpha: &Tensor, eps: f32) -> CResult<Tensor> {
    let eps_tensor = Tensor::full(eps as f64, xs.shape(), xs.device())?;
    let rms = xs
        .sqr()?
        .mean_keepdim(D::Minus1)?
        .broadcast_add(&eps_tensor)?
        .sqrt()?;
    let norm = xs.broadcast_div(&rms)?;
    norm.broadcast_mul(alpha)
}

/// Applies repetition penalty to logits.
///
/// This function modifies the logits based on previous tokens to penalize repetition.
///
/// # Arguments
///
/// * `logits` - The input logits tensor.
/// * `prev_tokens` - The previous tokens.
/// * `penalty` - The repetition penalty factor.
///
/// # Returns
///
/// * `Result<Tensor>` - The modified logits.
pub fn apply_repetition_penalty(
    logits: Tensor,
    prev_tokens: &[u32],
    penalty: f32,
) -> CResult<Tensor> {
    let mut logits_vec = logits.to_vec1::<f32>()?;
    let mut seen = HashSet::new();
    for &token in prev_tokens.iter().rev() {
        if seen.contains(&token) {
            continue;
        }
        seen.insert(token);
        if let Some(logit) = logits_vec.get_mut(token as usize) {
            if *logit >= 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
    Tensor::from_vec(logits_vec, logits.shape(), logits.device())
}

/// Checks if the tensor is all zeros.
///
/// This function checks if all elements in the tensor are zero.
///
/// # Arguments
///
/// * `tensor` - The input tensor.
///
/// # Returns
///
/// * `Result<bool>` - True if all elements are zero, false otherwise.
pub fn is_all_zero(tensor: &Tensor) -> CResult<bool> {
    let sum = tensor.sum_all()?;
    Ok(sum.to_scalar::<f32>()? == 0.0)
}

/// Clamps the tensor values between min and max.
///
/// This function clamps the tensor elements to the specified range.
///
/// # Arguments
///
/// * `tensor` - The input tensor.
/// * `min` - The minimum value.
/// * `max` - The maximum value.
///
/// # Returns
///
/// * `Result<Tensor>` - The clamped tensor.
pub fn clamp_tensor(tensor: &Tensor, min: f32, max: f32) -> CResult<Tensor> {
    let min_tensor = Tensor::new(min, tensor.device())?.broadcast_as(tensor.shape())?;
    let max_tensor = Tensor::new(max, tensor.device())?.broadcast_as(tensor.shape())?;
    tensor.maximum(&min_tensor)?.minimum(&max_tensor)
}

/// Device compatibility validator for safe tensor operations
pub struct DeviceValidator;

impl DeviceValidator {
    /// Create tensor with device validation and CPU fallback
    pub fn create_tensor_with_fallback(
        shape: impl Into<candle_core::Shape>,
        dtype: DType,
        preferred_device: &candle_core::Device,
    ) -> Result<Tensor, crate::error::MoshiError> {
        let shape = shape.into();
        
        // Try preferred device first
        match Tensor::zeros(shape.clone(), dtype, preferred_device) {
            Ok(tensor) => {
                tracing::debug!("Successfully created tensor on {:?}", preferred_device);
                Ok(tensor)
            }
            Err(device_err) => {
                tracing::warn!("Device {:?} failed, falling back to CPU: {}", preferred_device, device_err);
                
                // Fallback to CPU
                Tensor::zeros(shape.clone(), dtype, &candle_core::Device::Cpu)
                    .map_err(|cpu_err| crate::error::MoshiError::DeviceError(format!(
                        "Failed to create tensor on preferred device {:?} ({}) and CPU fallback also failed ({})",
                        preferred_device, device_err, cpu_err
                    )))
            }
        }
    }

    /// Create tensor with comprehensive error context
    pub fn create_empty_tensor(
        shape: impl Into<candle_core::Shape>,
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<Tensor, crate::error::MoshiError> {
        let shape = shape.into();
        Tensor::zeros(shape.clone(), dtype, device)
            .map_err(|e| crate::error::MoshiError::DeviceError(format!(
                "Failed to create tensor with shape {:?}, dtype {:?} on device {:?}: {}",
                shape, dtype, device, e
            )))
    }
}
