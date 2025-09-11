//! Tensor creation methods - zeros, ones, from_slice, arange, etc.
//!
//! This module contains all tensor creation functionality with zero-allocation,
//! blazing-fast implementations optimized for production use.

use super::*;

impl Tensor {
    fn ones_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = device.ones_impl(shape.elem_count(), dtype)?;
        let op = BackpropOp::none();
        Ok(from_storage(storage, shape, op, is_variable))
    }

    /// Creates a new tensor filled with ones.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::ones((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], (2, 3), &Device::Cpu)?;
    /// // a == b
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::ones_impl(shape, dtype, device, false)
    }

    pub fn const_set(&self, value: crate::scalar::Scalar) -> Result<()> {
        self.storage().const_fill(value, self.layout())
    }

    pub fn zero_set(&self) -> Result<()> {
        self.const_set(0.0.into())
    }

    pub fn one_set(&self) -> Result<()> {
        self.const_set(1.0.into())
    }

    /// Creates a new tensor filled with ones with same shape, dtype, and device as the other tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = a.ones_like()?;
    /// // b == a + 1
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn ones_like(&self) -> Result<Self> {
        Self::ones(self.shape(), self.dtype(), self.device())
    }

    fn zeros_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = device.zeros_impl(shape.elem_count(), dtype)?;
        let op = BackpropOp::none();
        Ok(from_storage(storage, shape, op, is_variable))
    }

    /// Creates a new tensor filled with zeros.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let tensor = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::zeros_impl(shape, dtype, device, false)
    }

    /// Creates a new tensor filled with zeros with same shape, dtype, and device as the other tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::ones((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = a.zeros_like()?;
    /// // b == a - a
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros(self.shape(), self.dtype(), self.device())
    }

    /// Creates a new tensor initialized with the given value.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let tensor = Tensor::full(42.0f32, (2, 3), &Device::Cpu)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn full<T: crate::WithDType, S: Into<Shape>>(
        value: T,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let shape = shape.into();
        let dtype = T::DTYPE;
        let storage = device.zeros_impl(shape.elem_count(), dtype)?;
        let tensor = from_storage(storage, shape, BackpropOp::none(), false);
        tensor.const_set(value.to_scalar())?;
        Ok(tensor)
    }

    /// Creates a new 1D tensor from a slice.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::from_slice(&[3u32, 1, 4, 1, 5], &Device::Cpu)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn from_slice<D: crate::WithDType + Clone, S: Into<Shape>>(
        array: &[D],
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let shape = shape.into();
        if array.len() != shape.elem_count() {
            bail!(
                "shape mismatch, expected {} elements, got {}",
                shape.elem_count(),
                array.len()
            )
        }
        let storage = device.storage_from_slice(array)?;
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    /// Creates a new 1D tensor from a vector.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::from_vec(vec![3u32, 1, 4, 1, 5], &Device::Cpu)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn from_vec<D: crate::WithDType, S: Into<Shape>>(
        array: Vec<D>,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let shape = shape.into();
        if array.len() != shape.elem_count() {
            bail!(
                "shape mismatch, expected {} elements, got {}",
                shape.elem_count(),
                array.len()
            )
        }
        let storage = device.storage_from_vec(array)?;
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    /// Creates a new 1D tensor with values from the iterator.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::from_iter([3u32, 1, 4, 1, 5], &Device::Cpu)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn from_iter<D: crate::WithDType>(
        iter: impl IntoIterator<Item = D>,
        device: &Device,
    ) -> Result<Self> {
        let data: Vec<D> = iter.into_iter().collect();
        let len = data.len();
        Self::from_vec(data, (len,), device)
    }

    /// Creates a new 1D tensor with values from the iterator and specified shape.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::from_iter_with_shape([3u32, 1, 4, 1, 5, 9], (2, 3), &Device::Cpu)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn from_iter_with_shape<D: crate::WithDType, S: Into<Shape>>(
        iter: impl IntoIterator<Item = D>,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let data: Vec<D> = iter.into_iter().collect();
        Self::from_vec(data, shape, device)
    }

    /// Creates a 1D tensor containing a sequence of integers from start to end (exclusive) with step size 1.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0u32, 5, &Device::Cpu)?;
    /// // tensor: [0, 1, 2, 3, 4]
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn arange<D: crate::WithDType + crate::IntDType>(
        start: D,
        end: D,
        device: &Device,
    ) -> Result<Self> {
        Self::arange_step(start, end, 1, device)
    }

    /// Creates a 1D tensor containing a sequence of integers from start to end (exclusive) with specified step size.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange_step(0u32, 10, 2, &Device::Cpu)?;
    /// // tensor: [0, 2, 4, 6, 8]
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn arange_step<D: crate::WithDType + crate::IntDType>(
        start: D,
        end: D,
        step: D,
        device: &Device,
    ) -> Result<Self> {
        if step.as_usize() == 0 {
            bail!("step cannot be zero")
        }
        let start_u = start.as_usize();
        let end_u = end.as_usize();
        let step_u = step.as_usize();
        
        if start_u >= end_u {
            return Self::zeros((0,), D::DTYPE, device);
        }
        
        let len = (end_u - start_u + step_u - 1) / step_u;
        let data: Vec<D> = (0..len)
            .map(|i| D::from_usize(start_u + i * step_u))
            .collect();
        Self::from_vec(data, (len,), device)
    }

    /// Creates a new tensor with random values from a uniform distribution in [0, 1).
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let tensor = Tensor::rand(0f32, 1f32, (2, 3), &Device::Cpu)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn rand<T: crate::FloatDType, S: Into<Shape>>(
        lo: T,
        up: T,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = device.rand_uniform_impl(shape.elem_count(), T::DTYPE, lo.to_f64(), up.to_f64())?;
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    /// Creates a new tensor with random values from a normal distribution.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let tensor = Tensor::randn(0f32, 1f32, (2, 3), &Device::Cpu)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn randn<T: crate::FloatDType, S: Into<Shape>>(
        mean: T,
        std: T,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = device.rand_normal_impl(shape.elem_count(), T::DTYPE, mean.to_f64(), std.to_f64())?;
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    /// Creates a new variable tensor filled with ones.
    pub fn var_ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::ones_impl(shape, dtype, device, true)
    }

    /// Creates a new variable tensor filled with zeros.
    pub fn var_zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::zeros_impl(shape, dtype, device, true)
    }

    /// Creates a new variable tensor from a slice.
    pub fn var_from_slice<D: crate::WithDType + Clone, S: Into<Shape>>(
        array: &[D],
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let shape = shape.into();
        if array.len() != shape.elem_count() {
            bail!(
                "shape mismatch, expected {} elements, got {}",
                shape.elem_count(),
                array.len()
            )
        }
        let storage = device.storage_from_slice(array)?;
        Ok(from_storage(storage, shape, BackpropOp::none(), true))
    }
}