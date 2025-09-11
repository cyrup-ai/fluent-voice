// src/nn.rs

use candle_core::{DType, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

pub type MaybeQuantizedEmbedding = Embedding;
pub type MaybeQuantizedLinear = Linear;
pub type MaybeQuantizedVarBuilder<'a> = VarBuilder<'a>;

pub fn linear<'a>(in_dim: usize, out_dim: usize, vb: VarBuilder<'a>) -> Result<Linear> {
    candle_nn::linear(in_dim, out_dim, vb)
}

pub fn linear_no_bias<'a>(in_dim: usize, out_dim: usize, vb: VarBuilder<'a>) -> Result<Linear> {
    candle_nn::linear_no_bias(in_dim, out_dim, vb)
}

pub fn matmul_dtype(lhs: &Tensor, rhs: &Tensor, dtype: Option<DType>) -> Result<Tensor> {
    match dtype {
        None => lhs.matmul(rhs),
        Some(dtype) => {
            let lhs = lhs.to_dtype(dtype)?;
            let rhs = rhs.to_dtype(dtype)?;
            let out = lhs.matmul(&rhs)?;
            out.to_dtype(lhs.dtype())
        }
    }
}

#[derive(Clone)]
pub struct SimpleLayer {
    w: Tensor,
    b: Option<Tensor>,
}

impl SimpleLayer {
    pub fn new(w: Tensor, b: Option<Tensor>) -> Self {
        Self { w, b }
    }
}

impl Module for SimpleLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let result = xs.matmul(&self.w)?;
        match &self.b {
            None => Ok(result),
            Some(b) => result.broadcast_add(b),
        }
    }
}
