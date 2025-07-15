// src/conditioner.rs

use candle::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};
use std::collections::HashMap;

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct Config {
    pub conditions: HashMap<String, (String, usize, usize, Vec<String>)>,
}

#[derive(Debug, Clone)]
pub struct LutConditioner {
    embed: Embedding,
    output_proj: Linear,
    _learnt_padding: Tensor,
    possible_values: HashMap<String, usize>,
}

impl LutConditioner {
    pub fn new(
        output_dim: usize,
        n_bins: usize,
        dim: usize,
        possible_values: Vec<String>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embed = candle_nn::embedding(n_bins + 1, dim, vb.pp("embed"))?;
        let output_proj = candle_nn::linear(dim, output_dim, vb.pp("output_proj"))?;
        let learnt_padding = vb.get((1, 1, output_dim), "learnt_padding")?;
        let possible_values_map: HashMap<String, usize> = possible_values
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        Ok(Self {
            embed,
            output_proj,
            _learnt_padding: learnt_padding,
            possible_values: possible_values_map,
        })
    }

    pub fn condition(&self, value: &str, device: &Device) -> Result<Tensor> {
        let idx = *self
            .possible_values
            .get(value)
            .ok_or_else(|| candle::Error::msg(format!("unknown value '{}'", value)))?;
        let cond = Tensor::from_slice(&[idx as u32], (1, 1), device)?
            .apply(&self.embed)?
            .apply(&self.output_proj)?;
        Ok(cond)
    }
}

#[derive(Debug, Clone)]
pub struct TensorConditioner {
    dim: usize,
}

impl TensorConditioner {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn condition(&self, tensor: Tensor) -> Result<Tensor> {
        if tensor.dim(candle::D::Minus1)? != self.dim {
            return Err(candle::Error::msg(
                "dimension mismatch in tensor conditioner",
            ));
        }
        Ok(tensor)
    }
}

#[derive(Debug, Clone)]
pub enum Conditioner {
    Lut(LutConditioner),
    Tensor(TensorConditioner),
}

#[derive(Debug, Clone)]
pub struct ConditionProvider {
    conditioners: HashMap<String, Conditioner>,
}

impl Conditioner {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        // For now, create a simple tensor conditioner
        // This can be expanded based on the config
        Ok(Conditioner::Tensor(TensorConditioner::new(512)))
    }
}

impl ConditionProvider {
    pub fn new(
        output_dim: usize,
        config: &HashMap<String, (String, usize, usize, Vec<String>)>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut conditioners = HashMap::with_capacity(config.len());
        for (name, (typ, n_bins, dim, possible_values)) in config {
            let conditioner_vb = vb.pp(name);
            let conditioner = match typ.as_str() {
                "lut" => Conditioner::Lut(LutConditioner::new(
                    output_dim,
                    *n_bins,
                    *dim,
                    possible_values.clone(),
                    conditioner_vb,
                )?),
                "tensor" => Conditioner::Tensor(TensorConditioner::new(*dim)),
                _ => {
                    return Err(candle::Error::msg(format!(
                        "unknown conditioner type '{}'",
                        typ
                    )));
                }
            };
            conditioners.insert(name.clone(), conditioner);
        }
        Ok(Self { conditioners })
    }

    pub fn get_condition(&self, name: &str, value: &str, device: &Device) -> Result<Tensor> {
        match self.conditioners.get(name) {
            Some(Conditioner::Lut(lut)) => lut.condition(value, device),
            Some(Conditioner::Tensor(tensor)) => {
                tensor.condition(Tensor::zeros((1, 1, tensor.dim), DType::F32, device)?)
            }
            None => Err(candle::Error::msg(format!(
                "unknown conditioner '{}'",
                name
            ))),
        }
    }
}

pub struct Condition {
    pub tensor: Tensor,
}

impl Condition {
    pub fn add_to_input(&self, input: &Tensor) -> Result<Tensor> {
        input.broadcast_add(&self.tensor)
    }
}
