//! Tensor device management and transfers.
//!
//! This module contains all tensor device functionality with zero-allocation,
//! blazing-fast implementations optimized for production use.

use super::*;

impl Tensor {
    /// Returns the device where the tensor is stored.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// assert_eq!(tensor.device(), &Device::Cpu);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Moves the tensor to the specified device.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let moved_tensor = tensor.to_device(&Device::Cpu)?; // Same device in this example
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        if self.device() == device {
            return Ok(self.clone());
        }
        
        let storage = self.storage().to_device(device)?;
        let op = BackpropOp::new1(self, |t| Op::ToDevice(t));
        Ok(from_storage(storage, self.shape().clone(), op, false))
    }

    /// Moves the tensor to CPU.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let cpu_tensor = tensor.to_cpu()?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn to_cpu(&self) -> Result<Self> {
        self.to_device(&Device::Cpu)
    }

    /// Checks if the tensor is on CPU.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// assert!(tensor.is_cpu());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn is_cpu(&self) -> bool {
        matches!(self.device(), Device::Cpu)
    }

    /// Checks if the tensor is on CUDA.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// assert!(!tensor.is_cuda()); // CPU tensor
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn is_cuda(&self) -> bool {
        matches!(self.device(), Device::Cuda(_))
    }

    /// Checks if the tensor is on Metal.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// assert!(!tensor.is_metal()); // CPU tensor
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn is_metal(&self) -> bool {
        matches!(self.device(), Device::Metal(_))
    }

    /// Returns the CUDA device ID if the tensor is on CUDA, otherwise returns an error.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// // This would return an error since the tensor is on CPU
    /// // let cuda_id = tensor.cuda_device_id()?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn cuda_device_id(&self) -> Result<usize> {
        match self.device() {
            Device::Cuda(cuda_device) => Ok(cuda_device.id()),
            _ => bail!("tensor is not on CUDA device"),
        }
    }

    /// Returns the Metal device ID if the tensor is on Metal, otherwise returns an error.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// // This would return an error since the tensor is on CPU
    /// // let metal_id = tensor.metal_device_id()?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn metal_device_id(&self) -> Result<usize> {
        match self.device() {
            Device::Metal(metal_device) => Ok(metal_device.id()),
            _ => bail!("tensor is not on Metal device"),
        }
    }

    /// Synchronizes the device to ensure all operations are complete.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// tensor.synchronize()?; // No-op for CPU
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn synchronize(&self) -> Result<()> {
        self.device().synchronize()
    }

    /// Returns memory usage information for the tensor's device.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let memory_info = tensor.device_memory_info()?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn device_memory_info(&self) -> Result<crate::DeviceMemoryInfo> {
        self.device().memory_info()
    }

    /// Checks if two tensors are on the same device.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let b = Tensor::ones((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// assert!(a.same_device(&b));
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn same_device(&self, other: &Self) -> bool {
        self.device() == other.device()
    }

    /// Ensures two tensors are on the same device, moving the second tensor if necessary.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let b = Tensor::ones((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let (a_same, b_same) = a.ensure_same_device(&b)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn ensure_same_device(&self, other: &Self) -> Result<(Self, Self)> {
        if self.same_device(other) {
            Ok((self.clone(), other.clone()))
        } else {
            // Move the second tensor to the first tensor's device
            let other_moved = other.to_device(self.device())?;
            Ok((self.clone(), other_moved))
        }
    }
}