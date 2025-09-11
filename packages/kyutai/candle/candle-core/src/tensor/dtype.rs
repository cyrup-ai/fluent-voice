//! Tensor data type handling and conversions.
//!
//! This module contains all tensor data type functionality with zero-allocation,
//! blazing-fast implementations optimized for production use.

use super::*;

impl Tensor {
    /// Converts the tensor to the specified data type.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let tensor = Tensor::arange(0u32, 5, &Device::Cpu)?;
    /// let float_tensor = tensor.to_dtype(DType::F32)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype() == dtype {
            return Ok(self.clone());
        }
        
        let storage = self.storage().to_dtype(self.layout(), dtype)?;
        let op = BackpropOp::new1(self, |t| Op::ToDevice(t));
        Ok(from_storage(storage, self.shape().clone(), op, false))
    }

    /// Converts the tensor to f32.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0u32, 5, &Device::Cpu)?;
    /// let float_tensor = tensor.to_f32()?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn to_f32(&self) -> Result<Self> {
        self.to_dtype(DType::F32)
    }

    /// Converts the tensor to f64.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0u32, 5, &Device::Cpu)?;
    /// let double_tensor = tensor.to_f64()?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn to_f64(&self) -> Result<Self> {
        self.to_dtype(DType::F64)
    }

    /// Converts the tensor to i32.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0f32, 5.0, &Device::Cpu)?;
    /// let int_tensor = tensor.to_i32()?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn to_i32(&self) -> Result<Self> {
        self.to_dtype(DType::I32)
    }

    /// Converts the tensor to i64.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0f32, 5.0, &Device::Cpu)?;
    /// let long_tensor = tensor.to_i64()?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn to_i64(&self) -> Result<Self> {
        self.to_dtype(DType::I64)
    }

    /// Converts the tensor to u32.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0f32, 5.0, &Device::Cpu)?;
    /// let uint_tensor = tensor.to_u32()?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn to_u32(&self) -> Result<Self> {
        self.to_dtype(DType::U32)
    }

    /// Converts the tensor to u8.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0f32, 5.0, &Device::Cpu)?;
    /// let byte_tensor = tensor.to_u8()?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn to_u8(&self) -> Result<Self> {
        self.to_dtype(DType::U8)
    }

    /// Returns the data type of the tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let tensor = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// assert_eq!(tensor.dtype(), DType::F32);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Checks if the tensor has a floating point data type.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let float_tensor = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let int_tensor = Tensor::zeros((2, 3), DType::I32, &Device::Cpu)?;
    /// assert!(float_tensor.is_floating_point());
    /// assert!(!int_tensor.is_floating_point());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn is_floating_point(&self) -> bool {
        matches!(self.dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    /// Checks if the tensor has an integer data type.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let float_tensor = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let int_tensor = Tensor::zeros((2, 3), DType::I32, &Device::Cpu)?;
    /// assert!(!float_tensor.is_integer());
    /// assert!(int_tensor.is_integer());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn is_integer(&self) -> bool {
        matches!(self.dtype, DType::U8 | DType::U32 | DType::I32 | DType::I64)
    }

    /// Checks if the tensor has a signed integer data type.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let signed_tensor = Tensor::zeros((2, 3), DType::I32, &Device::Cpu)?;
    /// let unsigned_tensor = Tensor::zeros((2, 3), DType::U32, &Device::Cpu)?;
    /// assert!(signed_tensor.is_signed_integer());
    /// assert!(!unsigned_tensor.is_signed_integer());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn is_signed_integer(&self) -> bool {
        matches!(self.dtype, DType::I32 | DType::I64)
    }

    /// Checks if the tensor has an unsigned integer data type.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let signed_tensor = Tensor::zeros((2, 3), DType::I32, &Device::Cpu)?;
    /// let unsigned_tensor = Tensor::zeros((2, 3), DType::U32, &Device::Cpu)?;
    /// assert!(!signed_tensor.is_unsigned_integer());
    /// assert!(unsigned_tensor.is_unsigned_integer());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn is_unsigned_integer(&self) -> bool {
        matches!(self.dtype, DType::U8 | DType::U32)
    }

    /// Returns the size in bytes of each element in the tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let f32_tensor = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let f64_tensor = Tensor::zeros((2, 3), DType::F64, &Device::Cpu)?;
    /// assert_eq!(f32_tensor.elem_size_in_bytes(), 4);
    /// assert_eq!(f64_tensor.elem_size_in_bytes(), 8);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn elem_size_in_bytes(&self) -> usize {
        self.dtype.size_in_bytes()
    }

    /// Returns the total size in bytes of the tensor data.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let tensor = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// assert_eq!(tensor.size_in_bytes(), 2 * 3 * 4); // 2*3 elements * 4 bytes each
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn size_in_bytes(&self) -> usize {
        self.shape().elem_count() * self.elem_size_in_bytes()
    }
}