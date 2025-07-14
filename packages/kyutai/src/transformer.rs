//! Transformer architecture components for Moshi language model

use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder, Module};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum PositionalEmbedding {
    None,
    Rope,
    Sinusoidal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum CrossAttentionGating {
    None,
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum NormType {
    LayerNorm,
    RmsNorm,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dim_feedforward: usize,
    pub causal: bool,
    pub norm_first: bool,
    pub bias_ff: bool,
    pub bias_attn: bool,
    pub layer_scale: Option<f64>,
    pub context: usize,
    pub max_period: usize,
    pub use_conv_block: bool,
    pub use_conv_bias: bool,
    pub cross_attention: Option<(CrossAttentionGating, NormType, Option<usize>)>,
    pub gating: Option<candle_nn::Activation>,
    pub norm: NormType,
    pub positional_embedding: PositionalEmbedding,
    pub conv_layout: bool,
    pub conv_kernel_size: usize,
    pub kv_repeat: usize,
    pub max_seq_len: usize,
    pub shared_cross_attn: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            d_model: 512,
            num_heads: 8,
            num_layers: 6,
            dim_feedforward: 2048,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 1024,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Relu),
            norm: NormType::LayerNorm,
            positional_embedding: PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        }
    }
}

#[derive(Debug)]
pub struct Norm {
    inner: NormInner,
}

#[derive(Debug)]
enum NormInner {
    LayerNorm(candle_nn::LayerNorm),
    RmsNorm(candle_nn::RmsNorm),
}

impl Norm {
    pub fn new(dim: usize, config: &Config, vb: VarBuilder) -> Result<Self> {
        let inner = match config.norm {
            NormType::LayerNorm => {
                NormInner::LayerNorm(candle_nn::layer_norm(dim, 1e-5, vb)?)
            }
            NormType::RmsNorm => {
                NormInner::RmsNorm(candle_nn::rms_norm(dim, 1e-5, vb)?)
            }
        };
        Ok(Self { inner })
    }
}

impl Module for Norm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.inner {
            NormInner::LayerNorm(ln) => ln.forward(xs),
            NormInner::RmsNorm(rms) => rms.forward(xs),
        }
    }
}

#[derive(Debug)]
pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MultiHeadAttention {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let d_model = config.d_model;
        let num_heads = config.num_heads;
        let head_dim = d_model / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        
        let q_proj = candle_nn::linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(d_model, d_model, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(d_model, d_model, vb.pp("out_proj"))?;
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale,
        })
    }
    
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let q = x.apply(&self.q_proj)?;
        let k = x.apply(&self.k_proj)?;
        let v = x.apply(&self.v_proj)?;
        
        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        
        let att = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;
        let att = match mask {
            Some(mask) => att.broadcast_add(mask)?,
            None => att,
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?;
        
        let out = out.transpose(1, 2)?.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
        out.apply(&self.out_proj)
    }
}

#[derive(Debug)]
pub struct FeedForward {
    w1: Linear,
    w2: Linear,
    activation: candle_nn::Activation,
}

impl FeedForward {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let d_model = config.d_model;
        let dim_feedforward = config.dim_feedforward;
        let activation = config.gating.unwrap_or(candle_nn::Activation::Relu);
        
        let w1 = candle_nn::linear(d_model, dim_feedforward, vb.pp("w1"))?;
        let w2 = candle_nn::linear(dim_feedforward, d_model, vb.pp("w2"))?;
        
        Ok(Self {
            w1,
            w2,
            activation,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.apply(&self.w1)?;
        let x = self.activation.forward(&x)?;
        x.apply(&self.w2)
    }
}

#[derive(Debug)]
pub struct TransformerLayer {
    self_attn: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: Norm,
    norm2: Norm,
    norm_first: bool,
}

impl TransformerLayer {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = MultiHeadAttention::new(config, vb.pp("self_attn"))?;
        let feed_forward = FeedForward::new(config, vb.pp("feed_forward"))?;
        let norm1 = Norm::new(config.d_model, config, vb.pp("norm1"))?;
        let norm2 = Norm::new(config.d_model, config, vb.pp("norm2"))?;
        
        Ok(Self {
            self_attn,
            feed_forward,
            norm1,
            norm2,
            norm_first: config.norm_first,
        })
    }
    
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = x.clone();
        
        let x = if self.norm_first {
            let x = x.apply(&self.norm1)?;
            let x = self.self_attn.forward(&x, mask)?;
            (residual + x)?
        } else {
            let x = self.self_attn.forward(x, mask)?;
            let x = (residual + x)?;
            x.apply(&self.norm1)?
        };
        
        let residual = x.clone();
        let x = if self.norm_first {
            let x = x.apply(&self.norm2)?;
            let x = self.feed_forward.forward(&x)?;
            (residual + x)?
        } else {
            let x = self.feed_forward.forward(&x)?;
            let x = (residual + x)?;
            x.apply(&self.norm2)?
        };
        
        Ok(x)
    }
}

#[derive(Debug)]
pub struct Transformer {
    pub layers: Vec<TransformerLayer>,
    pub d_model: usize,
}

impl Transformer {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        let vb_l = vb.pp("layers");
        for i in 0..config.num_layers {
            let layer = TransformerLayer::new(config, vb_l.pp(i))?;
            layers.push(layer);
        }
        
        Ok(Self {
            layers,
            d_model: config.d_model,
        })
    }
    
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x, mask)?;
        }
        Ok(x)
    }
}

// Utility function to create causal mask
pub fn create_causal_mask(seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_slice(&mask_data, (seq_len, seq_len), device)?.to_dtype(dtype)
}
