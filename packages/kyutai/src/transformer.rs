//! Transformer architecture components for Moshi language model

use crate::projected_transformer::TransformerCache;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder};
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

impl NormType {
    pub fn to_string(&self) -> &'static str {
        match self {
            NormType::LayerNorm => "layer_norm",
            NormType::RmsNorm => "rms_norm",
        }
    }
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

#[derive(Debug)]
struct FeedForward {
    w1: Linear,
    w2: Linear,
    activation: candle_nn::Activation,
}

impl FeedForward {
    pub fn new(
        d_model: usize,
        dim_feedforward: usize,
        bias: bool,
        activation: Option<candle_nn::Activation>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let w1 = if bias {
            candle_nn::linear(d_model, dim_feedforward, vb.pp("w1"))?
        } else {
            candle_nn::linear_no_bias(d_model, dim_feedforward, vb.pp("w1"))?
        };

        let w2 = if bias {
            candle_nn::linear(dim_feedforward, d_model, vb.pp("w2"))?
        } else {
            candle_nn::linear_no_bias(dim_feedforward, d_model, vb.pp("w2"))?
        };

        Ok(Self {
            w1,
            w2,
            activation: activation.unwrap_or(candle_nn::Activation::Relu),
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.w1)?.apply(&self.activation)?.apply(&self.w2)
    }
}

#[derive(Debug)]
pub struct TransformerLayer {
    self_attn: AttentionLayer,
    cross_attn: Option<AttentionLayer>,
    norm1: Norm,
    norm2: Norm,
    norm3: Option<Norm>,
    feed_forward: FeedForward,
    conv_block: Option<Conv1d>,
    layer_scale: Option<Tensor>,
}

impl TransformerLayer {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let d_model = config.d_model;

        let self_attn = AttentionLayer::new(
            d_model,
            config.num_heads,
            config.kv_repeat,
            config.bias_attn,
            config.causal,
            config.max_seq_len,
            vb.pp("self_attn"),
        )?;

        let cross_attn = if config.cross_attention.is_some() {
            Some(AttentionLayer::new(
                d_model,
                config.num_heads,
                config.kv_repeat,
                config.bias_attn,
                false, // cross attention is never causal
                config.max_seq_len,
                vb.pp("cross_attn"),
            )?)
        } else {
            None
        };

        let norm1 = Norm::new(d_model, &config.norm.to_string(), vb.pp("norm1"))?;
        let norm2 = Norm::new(d_model, &config.norm.to_string(), vb.pp("norm2"))?;
        let norm3 = if cross_attn.is_some() {
            Some(Norm::new(
                d_model,
                &config.norm.to_string(),
                vb.pp("norm3"),
            )?)
        } else {
            None
        };

        let feed_forward = FeedForward::new(
            d_model,
            config.dim_feedforward,
            config.bias_ff,
            config.gating,
            vb.pp("ff"),
        )?;

        let conv_block = if config.use_conv_block {
            let conv_config = Conv1dConfig {
                padding: config.conv_kernel_size / 2,
                ..Default::default()
            };
            Some(candle_nn::conv1d(
                d_model,
                d_model,
                config.conv_kernel_size,
                conv_config,
                vb.pp("conv"),
            )?)
        } else {
            None
        };

        let layer_scale = if let Some(scale) = config.layer_scale {
            Some(
                vb.get(d_model, "layer_scale")?
                    .broadcast_mul(&Tensor::new(&[scale], vb.device())?)?,
            )
        } else {
            None
        };

        Ok(Self {
            self_attn,
            cross_attn,
            norm1,
            norm2,
            norm3,
            feed_forward,
            conv_block,
            layer_scale,
        })
    }

    pub fn forward(&self, xs: &Tensor, _cross_attention_src: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;

        // Self-attention block
        let xs = self.norm1.forward(xs)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = if let Some(scale) = &self.layer_scale {
            xs.broadcast_mul(scale)?
        } else {
            xs
        };
        let xs = (xs + residual)?;

        // Feed-forward block
        let residual = &xs;
        let xs = self.norm2.forward(&xs)?;
        let xs = self.feed_forward.forward(&xs)?;
        let xs = if let Some(scale) = &self.layer_scale {
            xs.broadcast_mul(scale)?
        } else {
            xs
        };

        (xs + residual)
    }

    pub fn forward_with_cache(&self, xs: &Tensor, cache: &TransformerCache) -> Result<Tensor> {
        let residual = xs;

        // Self-attention block - normalized input
        let xs = self.norm1.forward(xs)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = if let Some(scale) = &self.layer_scale {
            xs.broadcast_mul(scale)?
        } else {
            xs
        };
        let xs = (xs + residual)?;

        // Cross-attention block (if configured)
        let xs = if let Some(cross_attn) = &self.cross_attn {
            if let Some(norm3) = &self.norm3 {
                let residual = &xs;
                let xs_norm = xs.apply(norm3)?;
                let cross_out = cross_attn.forward(&xs_norm)?;
                (residual + cross_out)?
            } else {
                xs
            }
        } else {
            xs
        };

        // Conv block (if configured)
        let xs = if let Some(conv_block) = &self.conv_block {
            let residual = &xs;
            // Conv1d expects (batch, channels, sequence) layout
            let xs_conv = xs.transpose(1, 2)?.apply(conv_block)?.transpose(1, 2)?;
            (residual + xs_conv)?
        } else {
            xs
        };

        // Feed-forward block
        let residual = &xs;
        let xs = self.norm2.forward(&xs)?;
        let xs = self.feed_forward.forward(&xs)?;
        let xs = if let Some(scale) = &self.layer_scale {
            xs.broadcast_mul(scale)?
        } else {
            xs
        };

        (xs + residual)
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
