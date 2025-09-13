//! GPU optimization utilities for Dia Voice
//!
//! This module provides essential device configuration and optimization
//! for maximum performance with Candle's built-in acceleration.

use crate::{DType, Device};

/// Configuration for GPU optimizations
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Enable mixed precision (F16/BF16) computation
    pub mixed_precision: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            mixed_precision: true,
        }
    }
}

/// Get optimized configuration for the device
pub fn get_optimal_config(device: &Device) -> GpuConfig {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => GpuConfig {
            mixed_precision: true,
        },
        #[cfg(feature = "metal")]
        Device::Metal(_) => GpuConfig {
            mixed_precision: true,
        },
        _ => GpuConfig::default(),
    }
}

/// Get optimal dtype for computation on the device
pub fn get_compute_dtype(device: &Device, mixed_precision: bool) -> DType {
    if !mixed_precision {
        return DType::F32;
    }

    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            // Use BF16 for modern NVIDIA GPUs (Ampere and newer)
            DType::BF16
        }
        #[cfg(feature = "metal")]
        Device::Metal(_) => {
            // Metal performs better with F16
            DType::F16
        }
        _ => DType::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_config_cpu() {
        let cpu_device = Device::Cpu;
        let config = get_optimal_config(&cpu_device);
        assert!(config.mixed_precision);
    }

    #[test]
    fn test_compute_dtype_no_mixed_precision() {
        let cpu_device = Device::Cpu;
        assert_eq!(get_compute_dtype(&cpu_device, false), DType::F32);
    }

    #[test]
    fn test_compute_dtype_mixed_precision_cpu() {
        let cpu_device = Device::Cpu;
        assert_eq!(get_compute_dtype(&cpu_device, true), DType::F32);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_compute_dtype_cuda() {
        // Note: This test will only run if CUDA is available and the feature is enabled
        let result = Device::new_cuda(0);
        if let Ok(cuda_device) = result {
            assert_eq!(get_compute_dtype(&cuda_device, true), DType::BF16);
            assert_eq!(get_compute_dtype(&cuda_device, false), DType::F32);
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_compute_dtype_metal() {
        // Note: This test will only run if Metal is available and the feature is enabled
        let result = Device::new_metal(0);
        if let Ok(metal_device) = result {
            assert_eq!(get_compute_dtype(&metal_device, true), DType::F16);
            assert_eq!(get_compute_dtype(&metal_device, false), DType::F32);
        }
    }

    #[test]
    fn test_gpu_config_default() {
        let default_config = GpuConfig::default();
        assert!(default_config.mixed_precision);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_optimal_config_cuda() {
        let result = Device::new_cuda(0);
        if let Ok(cuda_device) = result {
            let config = get_optimal_config(&cuda_device);
            assert!(config.mixed_precision);
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_optimal_config_metal() {
        let result = Device::new_metal(0);
        if let Ok(metal_device) = result {
            let config = get_optimal_config(&metal_device);
            assert!(config.mixed_precision);
        }
    }
}