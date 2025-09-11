//! Tensor module - Core tensor functionality decomposed into logical submodules.
//!
//! This module provides the main Tensor struct and coordinates all tensor operations
//! through specialized submodules for maximum maintainability and performance.

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BackpropOp, BinaryOp, CmpOp, Op, ReduceOp, UnaryOp};
use crate::scalar::TensorOrScalar;
use crate::shape::{Dim, Dims, ShapeWithOneHole};
use crate::{bail, storage::Storage, DType, Device, Error, Layout, Result, Shape};
use std::sync::{Arc, RwLock};

// Submodule declarations
pub mod creation;
pub mod operations;
pub mod indexing;
pub mod broadcasting;
pub mod dtype;
pub mod device;
pub mod storage;

// Re-export key types and functions from submodules
pub use creation::*;
pub use operations::*;
pub use indexing::*;
pub use broadcasting::*;
pub use dtype::*;
pub use device::*;
pub use storage::*;

/// Unique identifier for tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

impl TensorId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

pub struct Tensor_ {
    id: TensorId,
    // As we provide inner mutability on the tensor content, the alternatives are:
    // - Using a mutex, this would have the highest cost when retrieving the storage but would
    //   prevent errors when concurrent access takes place. Mutex would also be subject to
    //   deadlocks for example using the current code if the same tensor is used twice by a single
    //   binary op.
    // - Using a refcell unsafe cell would have some intermediary cost, borrow checking would be
    //   verified dynamically, but the resulting tensors would not be send or sync.
    // - Using an unsafe cell would have the lowest cost but undefined behavior on concurrent
    //   accesses.
    // Ideally, we would use Arc<Storage> for tensors on which we don't plan on modifying the data
    // and Arc<Mutex<Storage>> for tensors where the data could be modified, e.g. variables but
    // that's tricky to encode in the current setup.
    storage: Arc<RwLock<Storage>>,
    layout: Layout,
    op: BackpropOp,
    is_variable: bool,
    dtype: DType,
    device: Device,
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

// Tensors are refcounted so that cloning is cheap when building the op graph.
// Storages are also refcounted independently so that its possible to avoid
// copying the storage for operations that only modify the shape or stride.
#[derive(Clone)]
/// The core struct for manipulating tensors.
///
/// ```rust
/// use candle_core::{Tensor, DType, Device};
///
/// let a = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
/// let b = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((3, 4))?;
///
/// let c = a.matmul(&b)?;
/// # Ok::<(), candle_core::Error>(())
/// ```
///
/// Tensors are reference counted with [`Arc`] so cloning them is cheap.
pub struct Tensor(Arc<Tensor_>);

impl std::ops::Deref for Tensor {
    type Target = Tensor_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

/// Creates a fresh tensor structure based on a storage and a shape, this uses contiguous strides.
pub fn from_storage<S: Into<Shape>>(
    storage: Storage,
    shape: S,
    op: BackpropOp,
    is_variable: bool,
) -> Tensor {
    let shape = shape.into();
    let layout = Layout::contiguous(shape);
    let tensor_ = Tensor_ {
        id: TensorId::new(),
        storage: Arc::new(RwLock::new(storage)),
        layout,
        op,
        is_variable,
        dtype: storage.dtype(),
        device: storage.device().clone(),
    };
    Tensor(Arc::new(tensor_))
}