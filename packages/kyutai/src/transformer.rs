//! Transformer architecture components for Moshi language model

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
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
    ConditionalGatedSigmoid,
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
    pub layer_scale: Option<f32>,
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

#[derive(Debug, Clone)]
pub enum Norm {
    LayerNorm(candle_nn::LayerNorm),
    RmsNorm(candle_nn::RmsNorm),
}

impl Norm {
    pub fn new(d_model: usize, norm_type: &str, vb: VarBuilder) -> Result<Self> {
        match norm_type {
            "layer_norm" => {
                let ln = candle_nn::layer_norm(d_model, 1e-5, vb)?;
                Ok(Norm::LayerNorm(ln))
            }
            "rms_norm" => {
                let rms = candle_nn::rms_norm(d_model, 1e-5, vb)?;
                Ok(Norm::RmsNorm(rms))
            }
            _ => {
                let ln = candle_nn::layer_norm(d_model, 1e-5, vb)?;
                Ok(Norm::LayerNorm(ln))
            }
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Norm::LayerNorm(ln) => xs.apply(ln),
            Norm::RmsNorm(rms) => xs.apply(rms),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttentionLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    scale: f64,
    causal: bool,
    num_heads: usize,
    kv_repeat: usize,
    _max_seq_len: usize,
}

impl AttentionLayer {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        kv_repeat: usize,
        _bias_attn: bool,
        causal: bool,
        max_seq_len: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = d_model / num_heads;
        let q_proj = candle_nn::linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(d_model, num_heads * head_dim / kv_repeat, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(d_model, num_heads * head_dim / kv_repeat, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear(d_model, d_model, vb.pp("o_proj"))?;
        let scale = (head_dim as f64).powf(-0.5);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            scale,
            causal,
            num_heads,
            kv_repeat,
            _max_seq_len: max_seq_len,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, d_model) = xs.dims3()?;
        let head_dim = d_model / self.num_heads;

        let q = xs
            .apply(&self.q_proj)?
            .reshape((b_size, seq_len, self.num_heads, head_dim))?;
        let k = xs.apply(&self.k_proj)?.reshape((
            b_size,
            seq_len,
            self.num_heads / self.kv_repeat,
            head_dim,
        ))?;
        let v = xs.apply(&self.v_proj)?.reshape((
            b_size,
            seq_len,
            self.num_heads / self.kv_repeat,
            head_dim,
        ))?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let mut attn =
            (q.matmul(&k.transpose(candle::D::Minus2, candle::D::Minus1)?)? * self.scale)?;

        if self.causal {
            let mask = self.create_causal_mask(seq_len, xs.device())?;
            attn = attn.broadcast_add(&mask)?;
        }

        let attn = candle_nn::ops::softmax(&attn, candle::D::Minus1)?;
        let attn = attn.matmul(&v)?;
        let attn = attn.transpose(1, 2)?.reshape((b_size, seq_len, d_model))?;
        attn.apply(&self.o_proj)
    }

    fn create_causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let _mask = Tensor::ones((seq_len, seq_len), DType::F32, device)?;
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_slice(&mask_data, (seq_len, seq_len), device)
    }
}

#[derive(Debug, Clone)]
pub struct FFN {
    dense1: Linear,
    dense2: Linear,
    activation: candle_nn::Activation,
}

impl FFN {
    pub fn new(
        d_model: usize,
        dim_feedforward: usize,
        _bias_ff: bool,
        activation: candle_nn::Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dense1 = candle_nn::linear(d_model, dim_feedforward, vb.pp("dense1"))?;
        let dense2 = candle_nn::linear(dim_feedforward, d_model, vb.pp("dense2"))?;
        Ok(Self {
            dense1,
            dense2,
            activation,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.dense1)?
            .apply(&self.activation)?
            .apply(&self.dense2)
    }
}

#[derive(Debug, Clone)]
pub struct TransformerLayer {
    self_attn: AttentionLayer,
    ffn: FFN,
    norm1: Norm,
    norm2: Norm,
    _layer_scale: Option<f32>,
    norm_first: bool,
}

impl TransformerLayer {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = AttentionLayer::new(
            cfg.d_model,
            cfg.num_heads,
            cfg.kv_repeat,
            cfg.bias_attn,
            cfg.causal,
            cfg.max_seq_len,
            vb.pp("self_attn"),
        )?;
        let ffn = FFN::new(
            cfg.d_model,
            cfg.dim_feedforward,
            cfg.bias_ff,
            cfg.gating.unwrap_or(candle_nn::Activation::Silu),
            vb.pp("ffn"),
        )?;
        let norm1 = Norm::new(cfg.d_model, "layer_norm", vb.pp("norm1"))?;
        let norm2 = Norm::new(cfg.d_model, "layer_norm", vb.pp("norm2"))?;
        Ok(Self {
            self_attn,
            ffn,
            norm1,
            norm2,
            _layer_scale: cfg.layer_scale,
            norm_first: cfg.norm_first,
        })
    }

    pub fn forward(&self, xs: &Tensor, _cross_attention_src: Option<&Tensor>) -> Result<Tensor> {
        // Note: Cross-attention support not yet implemented in this layer
        // The _cross_attention_src parameter is accepted for API compatibility
        // TODO: Implement cross-attention mechanism when cross_attention config is enabled

        if self.norm_first {
            let normed = self.norm1.forward(xs)?;
            let attn = self.self_attn.forward(&normed)?;
            let xs = (xs + &attn)?;
            let normed = self.norm2.forward(&xs)?;
            let ffn = self.ffn.forward(&normed)?;
            xs + ffn
        } else {
            let attn = self.self_attn.forward(xs)?;
            let xs = (xs + &attn)?;
            let xs = self.norm1.forward(&xs)?;
            let ffn = self.ffn.forward(&xs)?;
            let xs = (&xs + &ffn)?;
            self.norm2.forward(&xs)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Transformer {
    layers: Vec<TransformerLayer>,
    norm: Norm,
}

impl Transformer {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for layer_idx in 0..cfg.num_layers {
            let layer = TransformerLayer::new(cfg, vb.pp(layer_idx))?;
            layers.push(layer);
        }
        let norm = Norm::new(cfg.d_model, "layer_norm", vb.pp("norm"))?;
        Ok(Self { layers, norm })
    }

    pub fn forward(&self, xs: &Tensor, cross_attention_src: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, cross_attention_src)?;
        }
        self.norm.forward(&xs)
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
