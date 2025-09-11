//! Tensor indexing and slicing operations.
//!
//! This module contains all tensor indexing functionality with zero-allocation,
//! blazing-fast implementations optimized for production use.

use super::*;

impl Tensor {
    /// Gets a single element from the tensor at the specified indices.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    /// let element = tensor.get(1)?; // Gets row 1
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn get(&self, index: usize) -> Result<Self> {
        let dims = self.shape().dims();
        if dims.is_empty() {
            bail!("cannot index a scalar tensor")
        }
        if index >= dims[0] {
            bail!("index {index} is out of bounds for dimension 0 with size {}", dims[0])
        }
        let mut new_dims = dims[1..].to_vec();
        if new_dims.is_empty() {
            new_dims.push(1);
        }
        let layout = self.layout().narrow(0, index, 1)?;
        let layout = layout.squeeze(0)?;
        let op = BackpropOp::new1(self, |t| Op::Narrow(t, 0, index, 1));
        Ok(from_storage(
            self.storage().clone(),
            Shape::from(new_dims),
            op,
            false,
        ))
    }

    /// Selects elements along an axis using an index tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    /// let indices = Tensor::from_slice(&[1u32, 0], &Device::Cpu)?;
    /// let result = tensor.index_select(&indices, 0)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn index_select(&self, indices: &Self, dim: usize) -> Result<Self> {
        let indices_shape = indices.shape();
        if indices_shape.rank() != 1 {
            bail!("index_select indices must be 1D, got shape {:?}", indices_shape)
        }
        if dim >= self.rank() {
            bail!("dimension {dim} is out of bounds for tensor with {rank} dimensions", rank = self.rank())
        }
        
        let mut new_dims = self.dims().to_vec();
        new_dims[dim] = indices_shape.dims()[0];
        let new_shape = Shape::from(new_dims);
        
        let storage = self.storage().index_select(indices.storage(), dim, self.layout(), indices.layout())?;
        let op = BackpropOp::new2(self, indices, |t, i| Op::IndexSelect(t, i, dim));
        Ok(from_storage(storage, new_shape, op, false))
    }

    /// Gathers elements along an axis using an index tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    /// let indices = Tensor::from_slice(&[1u32, 0, 2], &Device::Cpu)?;
    /// let result = tensor.gather(&indices, 1)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn gather(&self, indices: &Self, dim: usize) -> Result<Self> {
        if dim >= self.rank() {
            bail!("dimension {dim} is out of bounds for tensor with {rank} dimensions", rank = self.rank())
        }
        
        let storage = self.storage().gather(indices.storage(), dim, self.layout(), indices.layout())?;
        let op = BackpropOp::new2(self, indices, |t, i| Op::Gather(t, i, dim));
        Ok(from_storage(storage, indices.shape().clone(), op, false))
    }

    /// Scatters elements along an axis using an index tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::zeros((3, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let indices = Tensor::from_slice(&[0u32, 2], &Device::Cpu)?;
    /// let src = Tensor::ones((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let result = tensor.scatter_add(&indices, &src, 0)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn scatter_add(&self, indices: &Self, src: &Self, dim: usize) -> Result<Self> {
        if dim >= self.rank() {
            bail!("dimension {dim} is out of bounds for tensor with {rank} dimensions", rank = self.rank())
        }
        
        let storage = self.storage().scatter_add(
            indices.storage(),
            src.storage(),
            dim,
            self.layout(),
            indices.layout(),
            src.layout(),
        )?;
        let op = BackpropOp::new3(self, indices, src, |t, i, s| Op::ScatterAdd(t, i, s, dim));
        Ok(from_storage(storage, self.shape().clone(), op, false))
    }

    /// Slices the tensor along the specified dimension.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    /// let slice = tensor.narrow(1, 1, 2)?; // Slice columns 1-2
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        if dim >= self.rank() {
            bail!("dimension {dim} is out of bounds for tensor with {rank} dimensions", rank = self.rank())
        }
        if start + len > self.dim(dim)? {
            bail!(
                "narrow slice [{}:{}] is out of bounds for dimension {} with size {}",
                start,
                start + len,
                dim,
                self.dim(dim)?
            )
        }
        
        let layout = self.layout().narrow(dim, start, len)?;
        let op = BackpropOp::new1(self, |t| Op::Narrow(t, dim, start, len));
        Ok(from_storage(
            self.storage().clone(),
            layout.shape().clone(),
            op,
            false,
        ))
    }

    /// Chunks the tensor into the specified number of pieces along the given dimension.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    /// let chunks = tensor.chunk(3, 1)?; // Split into 3 chunks along dim 1
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Self>> {
        if dim >= self.rank() {
            bail!("dimension {dim} is out of bounds for tensor with {rank} dimensions", rank = self.rank())
        }
        if chunks == 0 {
            bail!("number of chunks must be positive")
        }
        
        let dim_size = self.dim(dim)?;
        let chunk_size = (dim_size + chunks - 1) / chunks; // Ceiling division
        let mut result = Vec::with_capacity(chunks);
        
        for i in 0..chunks {
            let start = i * chunk_size;
            if start >= dim_size {
                break;
            }
            let len = (chunk_size).min(dim_size - start);
            result.push(self.narrow(dim, start, len)?);
        }
        
        Ok(result)
    }

    /// Concatenates tensors along the specified dimension.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::ones((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let b = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let result = Tensor::cat(&[a, b], 0)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn cat(tensors: &[Self], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            bail!("cannot concatenate empty list of tensors")
        }
        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }
        
        let first = &tensors[0];
        if dim >= first.rank() {
            bail!("dimension {dim} is out of bounds for tensor with {rank} dimensions", rank = first.rank())
        }
        
        // Verify all tensors have compatible shapes
        let mut total_size = 0;
        for (i, tensor) in tensors.iter().enumerate() {
            if tensor.rank() != first.rank() {
                bail!("all tensors must have the same number of dimensions")
            }
            for d in 0..first.rank() {
                if d != dim && tensor.dim(d)? != first.dim(d)? {
                    bail!(
                        "tensor {} has incompatible shape at dimension {}: expected {}, got {}",
                        i,
                        d,
                        first.dim(d)?,
                        tensor.dim(d)?
                    )
                }
            }
            total_size += tensor.dim(dim)?;
        }
        
        let mut new_dims = first.dims().to_vec();
        new_dims[dim] = total_size;
        let new_shape = Shape::from(new_dims);
        
        let storages: Vec<_> = tensors.iter().map(|t| t.storage()).collect();
        let layouts: Vec<_> = tensors.iter().map(|t| t.layout()).collect();
        let storage = first.storage().cat(&storages[1..], &layouts, dim)?;
        
        let ops: Vec<_> = tensors.iter().map(|t| t.op.clone()).collect();
        let op = BackpropOp::cat(&ops, dim);
        Ok(from_storage(storage, new_shape, op, false))
    }

    /// Stacks tensors along a new dimension.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::ones((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let b = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;
    /// let result = Tensor::stack(&[a, b], 0)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn stack(tensors: &[Self], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            bail!("cannot stack empty list of tensors")
        }
        
        let first = &tensors[0];
        if dim > first.rank() {
            bail!("dimension {dim} is out of bounds for stacking")
        }
        
        // Verify all tensors have the same shape
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape() != first.shape() {
                bail!(
                    "tensor {} has incompatible shape: expected {:?}, got {:?}",
                    i,
                    first.shape(),
                    tensor.shape()
                )
            }
        }
        
        // Add new dimension to each tensor
        let unsqueezed: Result<Vec<_>> = tensors
            .iter()
            .map(|t| t.unsqueeze(dim))
            .collect();
        let unsqueezed = unsqueezed?;
        
        // Concatenate along the new dimension
        Self::cat(&unsqueezed, dim)
    }

    /// Flips the tensor along the specified dimensions.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    /// let flipped = tensor.flip(&[0, 1])?; // Flip both dimensions
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn flip(&self, dims: &[usize]) -> Result<Self> {
        let mut result = self.clone();
        for &dim in dims.iter() {
            if dim >= result.rank() {
                bail!("dimension {dim} is out of bounds for tensor with {rank} dimensions", rank = result.rank())
            }
            let size = result.dim(dim)?;
            let indices: Vec<i64> = (0..size).rev().map(|x| x as i64).collect();
            let indices_tensor = Tensor::from_vec(indices, (size,), result.device())?;
            result = result.index_select(&indices_tensor, dim)?;
        }
        Ok(result)
    }
}