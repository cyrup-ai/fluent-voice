# Optimize Inefficient Device Handling

## Problem
Tensor creation uses CPU device initially then requires transfer to target device, causing unnecessary data movement and performance overhead.

## Location
**File**: `packages/kyutai/src/speech_generator.rs`  
**Lines**: 1084

## Current Code
```rust
fn pcm_to_tensor(
    &self,
    samples: &[f32],
    config: &SpeakerPcmConfig,
) -> Result<candle_core::Tensor, SpeechGenerationError> {
    use candle_core::{DType, Device, Tensor};

    // Create tensor with shape [batch_size=1, channels, samples]
    let tensor = Tensor::from_vec(
        samples.to_vec(),
        (1, config.target_channels as usize, samples.len()),
        &Device::Cpu, // Use CPU for initial processing
    )
    .map_err(|e| {
        SpeechGenerationError::TensorOperation(format!("Failed to create tensor: {}", e))
    })?;
```

## Issue Details
- Hardcodes `Device::Cpu` for tensor creation
- Forces data to be created on CPU regardless of target device
- Requires expensive device transfer later if target is GPU/Metal
- Inefficient memory usage with temporary CPU allocation
- Performance bottleneck for GPU-accelerated inference

## Solution Required
1. **Create tensors directly on target device** from the generator's configuration
2. **Use the configured device** from `self.config.device`
3. **Eliminate unnecessary device transfers**
4. **Optimize memory allocation patterns**

## Implementation Steps
1. Replace hardcoded `&Device::Cpu` with `&self.config.device`
2. Ensure tensor creation uses the target device directly
3. Validate that the device is available before tensor creation
4. Add error handling for device-specific tensor creation failures
5. Test performance improvement with GPU/Metal devices

## Example Implementation
```rust
fn pcm_to_tensor(
    &self,
    samples: &[f32],
    config: &SpeakerPcmConfig,
) -> Result<candle_core::Tensor, SpeechGenerationError> {
    use candle_core::{DType, Device, Tensor};

    // Create tensor directly on target device
    let tensor = Tensor::from_vec(
        samples.to_vec(),
        (1, config.target_channels as usize, samples.len()),
        &self.config.device, // Use configured target device
    )
    .map_err(|e| {
        SpeechGenerationError::TensorOperation(format!(
            "Failed to create tensor on device {:?}: {}", 
            self.config.device, e
        ))
    })?;
```

## Priority
**LOW** - Performance optimization that doesn't affect functionality

## Acceptance Criteria
- [ ] Tensors created directly on target device from configuration
- [ ] No hardcoded CPU device usage
- [ ] Proper error messages include device information
- [ ] Performance improvement measurable with GPU/Metal devices
- [ ] No unnecessary device transfers in PCM processing pipeline