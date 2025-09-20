//! Transformer with input/output projections for Mimi codec

use crate::transformer::{Config as TransformerConfig, TransformerLayer};
use candle_core::{Result, Tensor};
use candle_nn::{Linear, VarBuilder};

/// Transformer with learnable input/output projections
#[derive(Debug)]
pub struct ProjectedTransformer {
    /// Input projection layer
    input_proj: Option<Linear>,
    /// Transformer layers
    layers: Vec<TransformerLayer>,
    /// Output projection layers (one per output)
    output_projs: Vec<Option<Linear>>,
    /// Configuration
    _config: TransformerConfig,
    /// Cache for streaming inference
    cache: Vec<TransformerCache>,
}

impl ProjectedTransformer {
    pub fn new(
        config: TransformerConfig,
        input_dim: usize,
        output_dims: Vec<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let d_model = config.d_model;

        // Create input projection if dimensions don't match
        let input_proj = if input_dim != d_model {
            Some(candle_nn::linear(input_dim, d_model, vb.pp("input_proj"))?)
        } else {
            None
        };

        // Create transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = TransformerLayer::new(&config, vb.pp(&format!("layers.{}", i)))?;
            layers.push(layer);
        }

        // Create output projections
        let mut output_projs = Vec::with_capacity(output_dims.len());
        for (i, &out_dim) in output_dims.iter().enumerate() {
            let proj = if out_dim != d_model {
                Some(candle_nn::linear(
                    d_model,
                    out_dim,
                    vb.pp(&format!("output_proj.{}", i)),
                )?)
            } else {
                None
            };
            output_projs.push(proj);
        }

        // Initialize cache for streaming
        let cache = layers.iter().map(|_| TransformerCache::new()).collect();

        Ok(Self {
            input_proj,
            layers,
            output_projs,
            _config: config,
            cache,
        })
    }

    pub fn forward(&self, xs: &Tensor, use_cache: bool) -> Result<Vec<Tensor>> {
        // Apply input projection
        let mut hidden = match &self.input_proj {
            Some(proj) => xs.apply(proj)?,
            None => xs.clone(),
        };

        // Apply transformer layers
        for (layer, cache) in self.layers.iter().zip(&self.cache) {
            hidden = if use_cache {
                layer.forward_with_cache(&hidden, cache)?
            } else {
                // Use cache-based forward to enable cross-attention support
                layer.forward_with_cache(&hidden, cache)?
            };
        }

        // Apply output projections
        let mut outputs = Vec::with_capacity(self.output_projs.len());
        for proj in &self.output_projs {
            let output = match proj {
                Some(p) => hidden.apply(p)?,
                None => hidden.clone(),
            };
            outputs.push(output);
        }

        Ok(outputs)
    }

    pub fn make_cache(&self) -> Vec<TransformerCache> {
        self.layers
            .iter()
            .map(|_| TransformerCache::new())
            .collect()
    }

    pub fn reset_cache(&mut self) {
        for cache in &mut self.cache {
            cache.reset();
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransformerCache {
    key_cache: Option<Tensor>,
    value_cache: Option<Tensor>,
    conv_state: Option<Tensor>,
}

impl TransformerCache {
    pub fn new() -> Self {
        Self {
            key_cache: None,
            value_cache: None,
            conv_state: None,
        }
    }

    pub fn reset(&mut self) {
        self.key_cache = None;
        self.value_cache = None;
        self.conv_state = None;
    }

    pub fn update_cache(&mut self, keys: Tensor, values: Tensor) -> Result<()> {
        self.key_cache = Some(match &self.key_cache {
            None => keys,
            Some(cached_keys) => Tensor::cat(&[cached_keys, &keys], 1)?,
        });

        self.value_cache = Some(match &self.value_cache {
            None => values,
            Some(cached_values) => Tensor::cat(&[cached_values, &values], 1)?,
        });

        Ok(())
    }

    pub fn get_cached_kv(&self) -> (Option<&Tensor>, Option<&Tensor>) {
        (self.key_cache.as_ref(), self.value_cache.as_ref())
    }
}
