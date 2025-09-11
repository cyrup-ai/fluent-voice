//! Tensor mathematical operations - add, mul, matmul, etc.
//!
//! This module contains all tensor mathematical operations with zero-allocation,
//! blazing-fast implementations optimized for production use.

use super::*;

// Macro for unary operations
macro_rules! unary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self) -> Result<Self> {
            let shape = self.shape();
            if shape.elem_count() == 0 {
                return Ok(self.clone());
            }
            let storage = self
                .storage()
                .unary_impl::<crate::op::$op_name>(self.layout())?;
            let op = BackpropOp::new1(self, |s| Op::Unary(s, UnaryOp::$op_name));
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    };
}

// Macro for binary operations
macro_rules! binary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let shape = self.same_shape_binary_op(rhs, stringify!($fn_name))?;
            if shape.elem_count() == 0 {
                return Ok(self.clone());
            }
            let storage = self.storage().binary_impl::<crate::op::$op_name>(
                &*rhs.storage(),
                self.layout(),
                rhs.layout(),
            )?;
            let op = BackpropOp::new2(self, rhs, |t1, t2| Op::Binary(t1, t2, BinaryOp::$op_name));
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    };
}

// Macro for binary operations with scalar support
macro_rules! binary_op_scalar {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name<T: TensorOrScalar>(&self, rhs: T) -> Result<Self> {
            let rhs = match rhs.to_tensor_scalar()? {
                crate::scalar::TensorScalar::Tensor(t) => t,
                crate::scalar::TensorScalar::Scalar(s) => {
                    Tensor::full(s, self.shape(), self.device())?
                }
            };
            let shape = self.same_shape_binary_op(&rhs, stringify!($fn_name))?;
            if shape.elem_count() == 0 {
                return Ok(self.clone());
            }
            let storage = self.storage().binary_impl::<crate::op::$op_name>(
                &*rhs.storage(),
                self.layout(),
                rhs.layout(),
            )?;
            let op = BackpropOp::new2(self, &rhs, |t1, t2| Op::Binary(t1, t2, BinaryOp::$op_name));
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    };
}

// Macro for broadcast binary operations
macro_rules! broadcast_binary_op {
    ($fn_name:ident, $inner_fn_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let lhs = self;
            let shape = lhs
                .shape()
                .broadcast_shape_binary_op(rhs.shape(), stringify!($fn_name))?;
            let lhs_broadcast = lhs.shape() != &shape;
            let rhs_broadcast = rhs.shape() != &shape;
            match (lhs_broadcast, rhs_broadcast) {
                (true, true) => lhs.broadcast_as(&shape)?.$inner_fn_name(&rhs.broadcast_as(&shape)?),
                (false, true) => lhs.$inner_fn_name(&rhs.broadcast_as(&shape)?),
                (true, false) => lhs.broadcast_as(&shape)?.$inner_fn_name(rhs),
                (false, false) => lhs.$inner_fn_name(rhs),
            }
        }
    };
}

impl Tensor {
    // Unary operations
    unary_op!(neg, Neg);
    unary_op!(recip, Recip);
    unary_op!(exp, Exp);
    unary_op!(log, Log);
    unary_op!(sin, Sin);
    unary_op!(cos, Cos);
    unary_op!(abs, Abs);
    unary_op!(sqr, Sqr);
    unary_op!(sqrt, Sqrt);
    unary_op!(gelu, Gelu);
    unary_op!(erf, Erf);
    unary_op!(relu, Relu);
    unary_op!(silu, Silu);

    // Binary operations
    binary_op!(add_, Add);
    binary_op!(sub_, Sub);
    binary_op!(mul_, Mul);
    binary_op!(div_, Div);
    binary_op!(minimum, Minimum);
    binary_op!(maximum, Maximum);

    // Binary operations with scalar support
    binary_op_scalar!(add, Add);
    binary_op_scalar!(sub, Sub);
    binary_op_scalar!(mul, Mul);
    binary_op_scalar!(div, Div);

    // Broadcast binary operations
    broadcast_binary_op!(broadcast_add, add_);
    broadcast_binary_op!(broadcast_sub, sub_);
    broadcast_binary_op!(broadcast_mul, mul_);
    broadcast_binary_op!(broadcast_div, div_);
    broadcast_binary_op!(broadcast_minimum, minimum);
    broadcast_binary_op!(broadcast_maximum, maximum);

    /// Matrix multiplication.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    /// let b = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((3, 4))?;
    /// let c = a.matmul(&b)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let (lhs_l, lhs_r) = self.dims2()?;
        let (rhs_l, rhs_r) = rhs.dims2()?;
        if lhs_r != rhs_l {
            bail!(
                "matmul dimension mismatch: lhs: {lhs_l}x{lhs_r}, rhs: {rhs_l}x{rhs_r}"
            )
        }
        let shape = Shape::from((lhs_l, rhs_r));
        let storage = self.storage().matmul(
            &*rhs.storage(),
            (lhs_l, lhs_r, rhs_r),
            self.layout(),
            rhs.layout(),
        )?;
        let op = BackpropOp::new2(self, rhs, |t1, t2| Op::MatMul(t1, t2));
        Ok(from_storage(storage, shape, op, false))
    }

    /// Affine transformation: `self * mul + add`.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0f32, 6f32, &Device::Cpu)?;
    /// let result = tensor.affine(2.0, 1.0)?; // 2*x + 1
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> {
        let storage = self.storage().affine(self.layout(), mul, add)?;
        let op = BackpropOp::new1(self, |t| Op::Affine { mul, add });
        Ok(from_storage(storage, self.shape().clone(), op, false))
    }

    /// Element-wise power operation.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(1f32, 5f32, &Device::Cpu)?;
    /// let result = tensor.powf(2.0)?; // x^2
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn powf(&self, exponent: f64) -> Result<Self> {
        let storage = self.storage().powf(self.layout(), exponent)?;
        let op = BackpropOp::new1(self, |t| Op::Powf(t, exponent));
        Ok(from_storage(storage, self.shape().clone(), op, false))
    }

    /// Element-wise power operation with tensor exponent.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let base = Tensor::arange(1f32, 5f32, &Device::Cpu)?;
    /// let exp = Tensor::full(2.0f32, base.shape(), &Device::Cpu)?;
    /// let result = base.pow(&exp)?; // base^exp
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn pow(&self, exponent: &Self) -> Result<Self> {
        let shape = self.same_shape_binary_op(exponent, "pow")?;
        if shape.elem_count() == 0 {
            return Ok(self.clone());
        }
        let storage = self.storage().binary_impl::<crate::op::Pow>(
            &*exponent.storage(),
            self.layout(),
            exponent.layout(),
        )?;
        let op = BackpropOp::new2(self, exponent, |t1, t2| Op::Binary(t1, t2, BinaryOp::Pow));
        Ok(from_storage(storage, shape.clone(), op, false))
    }

    /// Clamps all elements to be within [min, max].
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(-2f32, 3f32, &Device::Cpu)?;
    /// let result = tensor.clamp(-1.0, 1.0)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn clamp<T: crate::FloatDType>(&self, min: T, max: T) -> Result<Self> {
        let min_t = Tensor::full(min, self.shape(), self.device())?;
        let max_t = Tensor::full(max, self.shape(), self.device())?;
        self.maximum(&min_t)?.minimum(&max_t)
    }

    /// Returns the sum of all elements.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(1f32, 5f32, &Device::Cpu)?;
    /// let sum = tensor.sum_all()?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn sum_all(&self) -> Result<Self> {
        let shape = Shape::from(());
        let storage = self.storage().reduce_op(ReduceOp::Sum, self.layout(), &[0, 1, 2, 3])?;
        let op = BackpropOp::new1(self, |t| Op::Reduce(t, ReduceOp::Sum));
        Ok(from_storage(storage, shape, op, false))
    }

    /// Returns the mean of all elements.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(1f32, 5f32, &Device::Cpu)?;
    /// let mean = tensor.mean_all()?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn mean_all(&self) -> Result<Self> {
        let sum = self.sum_all()?;
        let elem_count = self.shape().elem_count() as f64;
        sum.affine(1.0 / elem_count, 0.0)
    }

    /// Applies softmax along the last dimension.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    /// let result = tensor.softmax(candle_core::D::Minus1)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn softmax<D: Dim>(&self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "softmax")?;
        let max_val = self.max_keepdim(dim)?;
        let diff = self.broadcast_sub(&max_val)?;
        let exp_diff = diff.exp()?;
        let sum_exp = exp_diff.sum_keepdim(dim)?;
        exp_diff.broadcast_div(&sum_exp)
    }

    /// Applies log softmax along the specified dimension.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    /// let result = tensor.log_softmax(candle_core::D::Minus1)?;
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn log_softmax<D: Dim>(&self, dim: D) -> Result<Self> {
        let softmax = self.softmax(dim)?;
        softmax.log()
    }
}