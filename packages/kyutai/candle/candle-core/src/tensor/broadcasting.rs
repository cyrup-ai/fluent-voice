//! Tensor broadcasting logic and utilities.
//!
//! This module contains all tensor broadcasting functionality with zero-allocation,
//! blazing-fast implementations optimized for production use.

use super::*;

impl Tensor {
    /// Broadcasts the tensor to the specified shape.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device, Shape};
    /// let tensor = Tensor::ones((1, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let broadcasted = tensor.broadcast_as(&Shape::from((2, 3)))?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let target_shape = shape.into();
        if self.shape() == &target_shape {
            return Ok(self.clone());
        }
        
        let layout = self.layout().broadcast_as(&target_shape)?;
        let op = BackpropOp::new1(self, |t| Op::Broadcast(t));
        Ok(from_storage(
            self.storage().clone(),
            target_shape,
            op,
            false,
        ))
    }

    /// Broadcasts the tensor to be compatible with another tensor's shape.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::ones((1, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let b = Tensor::ones((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let broadcasted = a.broadcast_to(&b)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn broadcast_to(&self, other: &Self) -> Result<Self> {
        self.broadcast_as(other.shape())
    }

    /// Broadcasts tensors to a common shape for binary operations.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::ones((1, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let b = Tensor::ones((2, 1), candle_core::DType::F32, &Device::Cpu)?;
    /// let (a_bc, b_bc) = Tensor::broadcast_tensors(&a, &b)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn broadcast_tensors(lhs: &Self, rhs: &Self) -> Result<(Self, Self)> {
        let shape = lhs
            .shape()
            .broadcast_shape_binary_op(rhs.shape(), "broadcast_tensors")?;
        let lhs_bc = if lhs.shape() == &shape {
            lhs.clone()
        } else {
            lhs.broadcast_as(&shape)?
        };
        let rhs_bc = if rhs.shape() == &shape {
            rhs.clone()
        } else {
            rhs.broadcast_as(&shape)?
        };
        Ok((lhs_bc, rhs_bc))
    }

    /// Expands the tensor by adding singleton dimensions.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::ones((3,), candle_core::DType::F32, &Device::Cpu)?;
    /// let expanded = tensor.expand(&[1, 3, 1])?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let target_shape = shape.into();
        if self.shape().rank() > target_shape.rank() {
            bail!(
                "expand: target shape rank {} is less than source shape rank {}",
                target_shape.rank(),
                self.shape().rank()
            )
        }
        
        // Check that dimensions are compatible
        let self_dims = self.shape().dims();
        let target_dims = target_shape.dims();
        let offset = target_dims.len() - self_dims.len();
        
        for (i, &self_dim) in self_dims.iter().enumerate() {
            let target_dim = target_dims[i + offset];
            if self_dim != 1 && self_dim != target_dim {
                bail!(
                    "expand: dimension {} cannot be expanded from {} to {}",
                    i,
                    self_dim,
                    target_dim
                )
            }
        }
        
        self.broadcast_as(target_shape)
    }

    /// Repeats the tensor along specified dimensions.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::ones((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let repeated = tensor.repeat(&[2, 1])?; // Repeat 2x along dim 0
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn repeat(&self, repeats: &[usize]) -> Result<Self> {
        if repeats.len() != self.rank() {
            bail!(
                "repeat: number of repeats {} must match tensor rank {}",
                repeats.len(),
                self.rank()
            )
        }
        
        let mut result = self.clone();
        for (dim, &repeat_count) in repeats.iter().enumerate() {
            if repeat_count == 1 {
                continue;
            }
            if repeat_count == 0 {
                bail!("repeat count cannot be zero")
            }
            
            // Create a list of the tensor repeated along this dimension
            let tensors: Vec<_> = (0..repeat_count).map(|_| result.clone()).collect();
            result = Self::cat(&tensors, dim)?;
        }
        
        Ok(result)
    }

    /// Tiles the tensor by repeating it along each dimension.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::ones((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let tiled = tensor.tile(&[2, 1])?; // Same as repeat
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn tile(&self, repeats: &[usize]) -> Result<Self> {
        self.repeat(repeats)
    }

    /// Broadcasts and performs element-wise addition with automatic shape alignment.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::ones((1, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let b = Tensor::ones((2, 1), candle_core::DType::F32, &Device::Cpu)?;
    /// let result = a.broadcast_add(&b)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        let (lhs_bc, rhs_bc) = Self::broadcast_tensors(self, rhs)?;
        lhs_bc.add_(&rhs_bc)
    }

    /// Broadcasts and performs element-wise subtraction with automatic shape alignment.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::ones((1, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let b = Tensor::ones((2, 1), candle_core::DType::F32, &Device::Cpu)?;
    /// let result = a.broadcast_sub(&b)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn broadcast_sub(&self, rhs: &Self) -> Result<Self> {
        let (lhs_bc, rhs_bc) = Self::broadcast_tensors(self, rhs)?;
        lhs_bc.sub_(&rhs_bc)
    }

    /// Broadcasts and performs element-wise multiplication with automatic shape alignment.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::ones((1, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let b = Tensor::ones((2, 1), candle_core::DType::F32, &Device::Cpu)?;
    /// let result = a.broadcast_mul(&b)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn broadcast_mul(&self, rhs: &Self) -> Result<Self> {
        let (lhs_bc, rhs_bc) = Self::broadcast_tensors(self, rhs)?;
        lhs_bc.mul_(&rhs_bc)
    }

    /// Broadcasts and performs element-wise division with automatic shape alignment.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::ones((1, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let b = Tensor::ones((2, 1), candle_core::DType::F32, &Device::Cpu)?;
    /// let result = a.broadcast_div(&b)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn broadcast_div(&self, rhs: &Self) -> Result<Self> {
        let (lhs_bc, rhs_bc) = Self::broadcast_tensors(self, rhs)?;
        lhs_bc.div_(&rhs_bc)
    }

    /// Broadcasts and performs element-wise minimum with automatic shape alignment.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::ones((1, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let b = Tensor::zeros((2, 1), candle_core::DType::F32, &Device::Cpu)?;
    /// let result = a.broadcast_minimum(&b)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn broadcast_minimum(&self, rhs: &Self) -> Result<Self> {
        let (lhs_bc, rhs_bc) = Self::broadcast_tensors(self, rhs)?;
        lhs_bc.minimum(&rhs_bc)
    }

    /// Broadcasts and performs element-wise maximum with automatic shape alignment.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::ones((1, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let b = Tensor::zeros((2, 1), candle_core::DType::F32, &Device::Cpu)?;
    /// let result = a.broadcast_maximum(&b)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn broadcast_maximum(&self, rhs: &Self) -> Result<Self> {
        let (lhs_bc, rhs_bc) = Self::broadcast_tensors(self, rhs)?;
        lhs_bc.maximum(&rhs_bc)
    }
}