//! Tensor underlying storage management.
//!
//! This module contains all tensor storage functionality with zero-allocation,
//! blazing-fast implementations optimized for production use.

use super::*;

impl Tensor {
    /// Returns a reference to the underlying storage.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let storage = tensor.storage();
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn storage(&self) -> Arc<RwLock<Storage>> {
        self.storage.clone()
    }

    /// Returns the layout of the tensor (shape and strides).
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let layout = tensor.layout();
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns the shape of the tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let shape = tensor.shape();
    /// assert_eq!(shape.dims(), &[2, 3]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    /// Returns the strides of the tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let strides = tensor.stride();
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn stride(&self) -> &[usize] {
        self.layout.stride()
    }

    /// Returns the number of dimensions (rank) of the tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3, 4), candle_core::DType::F32, &Device::Cpu)?;
    /// assert_eq!(tensor.rank(), 3);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    /// Returns the dimensions of the tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3, 4), candle_core::DType::F32, &Device::Cpu)?;
    /// assert_eq!(tensor.dims(), &[2, 3, 4]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    /// Returns the size of the specified dimension.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3, 4), candle_core::DType::F32, &Device::Cpu)?;
    /// assert_eq!(tensor.dim(0)?, 2);
    /// assert_eq!(tensor.dim(1)?, 3);
    /// assert_eq!(tensor.dim(2)?, 4);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn dim<D: Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(self.shape(), "dim")?;
        Ok(self.dims()[dim])
    }

    /// Returns the dimensions as a 2-tuple for 2D tensors.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let (rows, cols) = tensor.dims2()?;
    /// assert_eq!(rows, 2);
    /// assert_eq!(cols, 3);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn dims2(&self) -> Result<(usize, usize)> {
        let dims = self.dims();
        if dims.len() != 2 {
            bail!("dims2 can only be called on 2D tensors, got shape {:?}", self.shape())
        }
        Ok((dims[0], dims[1]))
    }

    /// Returns the dimensions as a 3-tuple for 3D tensors.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3, 4), candle_core::DType::F32, &Device::Cpu)?;
    /// let (d0, d1, d2) = tensor.dims3()?;
    /// assert_eq!(d0, 2);
    /// assert_eq!(d1, 3);
    /// assert_eq!(d2, 4);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        let dims = self.dims();
        if dims.len() != 3 {
            bail!("dims3 can only be called on 3D tensors, got shape {:?}", self.shape())
        }
        Ok((dims[0], dims[1], dims[2]))
    }

    /// Returns the dimensions as a 4-tuple for 4D tensors.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3, 4, 5), candle_core::DType::F32, &Device::Cpu)?;
    /// let (d0, d1, d2, d3) = tensor.dims4()?;
    /// assert_eq!(d0, 2);
    /// assert_eq!(d1, 3);
    /// assert_eq!(d2, 4);
    /// assert_eq!(d3, 5);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        let dims = self.dims();
        if dims.len() != 4 {
            bail!("dims4 can only be called on 4D tensors, got shape {:?}", self.shape())
        }
        Ok((dims[0], dims[1], dims[2], dims[3]))
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3, 4), candle_core::DType::F32, &Device::Cpu)?;
    /// assert_eq!(tensor.elem_count(), 2 * 3 * 4);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn elem_count(&self) -> usize {
        self.shape().elem_count()
    }

    /// Checks if the tensor is contiguous in memory.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// assert!(tensor.is_contiguous());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Returns a contiguous version of the tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let contiguous = tensor.contiguous()?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            let storage = self.storage().copy_strided_src(&self.layout, 0)?;
            let op = BackpropOp::new1(self, |t| Op::Copy(t));
            Ok(from_storage(storage, self.shape().clone(), op, false))
        }
    }

    /// Checks if the tensor is a scalar (0-dimensional).
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let scalar = Tensor::from_slice(&[42.0f32], (), &Device::Cpu)?;
    /// let vector = Tensor::zeros((3,), candle_core::DType::F32, &Device::Cpu)?;
    /// assert!(scalar.is_scalar());
    /// assert!(!vector.is_scalar());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }

    /// Checks if the tensor is a vector (1-dimensional).
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let vector = Tensor::zeros((3,), candle_core::DType::F32, &Device::Cpu)?;
    /// let matrix = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// assert!(vector.is_vector());
    /// assert!(!matrix.is_vector());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn is_vector(&self) -> bool {
        self.rank() == 1
    }

    /// Checks if the tensor is a matrix (2-dimensional).
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let matrix = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let tensor3d = Tensor::zeros((2, 3, 4), candle_core::DType::F32, &Device::Cpu)?;
    /// assert!(matrix.is_matrix());
    /// assert!(!tensor3d.is_matrix());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn is_matrix(&self) -> bool {
        self.rank() == 2
    }

    /// Returns the unique identifier for this tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let id = tensor.id();
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Checks if this tensor is a variable (requires gradients).
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let var_tensor = Tensor::var_zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// assert!(!tensor.is_variable());
    /// assert!(var_tensor.is_variable());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn is_variable(&self) -> bool {
        self.is_variable
    }

    /// Returns the backpropagation operation for this tensor.
    pub fn op(&self) -> &BackpropOp {
        &self.op
    }

    /// Detaches the tensor from the computation graph.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let var_tensor = Tensor::var_zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let detached = var_tensor.detach();
    /// assert!(!detached.is_variable());
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn detach(&self) -> Self {
        if !self.is_variable {
            self.clone()
        } else {
            from_storage(
                self.storage().clone(),
                self.shape().clone(),
                BackpropOp::none(),
                false,
            )
        }
    }
}