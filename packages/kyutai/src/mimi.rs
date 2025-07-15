// use super::NormType; // TODO: uncomment when audio module is available
// use super::streaming::StreamingModule; // TODO: uncomment when needed
use super::transformer;
// use super::{conv, nn, quantization, seanet}; // TODO: uncomment when modules are implemented
// use candle::{DType, Device, Module, Result, Tensor}; // TODO: uncomment when needed
// use candle_nn::VarBuilder; // TODO: uncomment when needed

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ResampleMethod {
    Conv,
    Interpolate,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub channels: usize,
    pub sample_rate: f64,
    pub frame_rate: f64,
    pub renormalize: bool,
    pub resample_method: ResampleMethod,
    // pub seanet: seanet::Config, // TODO: uncomment when seanet module is implemented
    pub transformer: transformer::Config,
    pub quantizer_n_q: usize,
    pub quantizer_bins: usize,
    pub quantizer_dim: usize,
}

impl Config {
    // TODO: implement v0_1 method when seanet and conv modules are available
    /*
    // /lustre/scwpod02/client/kyutai/alex/mimi_exp/xps/b7d2bd5a/.hydra/config.yaml
    pub fn v0_1(num_codebooks: Option<usize>) -> Self {
        let seanet_cfg = seanet::Config {
            dimension: 512,
            channels: 1,
            causal: true,
            n_filters: 64,
            n_residual_layers: 1,
            activation: candle_nn::Activation::Elu(1.),
            compress: 2,
            dilation_base: 2,
            disable_norm_outer_blocks: 0,
            final_activation: None,
            kernel_size: 7,
            residual_kernel_size: 3,
            last_kernel_size: 3,
            lstm: 0,
            norm: conv::Norm::WeightNorm,
            pad_mode: conv::PadMode::Constant,
            ratios: vec![8, 6, 5, 4],
            true_skip: true,
        };
        let transformer_cfg = transformer::Config {
            d_model: seanet_cfg.dimension,
            num_heads: 8,
            num_layers: 8,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: Some(0.01),
            context: 250,
            conv_kernel_size: 5,
            use_conv_bias: true,
            use_conv_block: false,
            cross_attention: None,
            max_period: 10000,
            gating: None,
            norm: NormType::LayerNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,

            dim_feedforward: 2048,
            kv_repeat: 1,
            conv_layout: true, // see builders.py
            max_seq_len: 8192, // the transformer works at 25hz so this is ~5 mins.
            shared_cross_attn: false,
        };
        Config {
            channels: 1,
            sample_rate: 24_000.,
            frame_rate: 12.5,
            renormalize: true,
            resample_method: ResampleMethod::Conv,
            seanet: seanet_cfg,
            transformer: transformer_cfg,
            quantizer_n_q: num_codebooks.unwrap_or(16),
            quantizer_bins: 2048,
            quantizer_dim: 256,
        }
    }
    */
}

// Minimal Mimi struct stub to satisfy interface requirements
#[derive(Debug, Clone)]
pub struct Mimi {
    config: Config,
    device: candle::Device,
}

// Minimal Mimi implementation stub to satisfy interface requirements
impl Mimi {
    pub fn new(cfg: Config, _vb: candle_nn::VarBuilder) -> candle::Result<Self> {
        Ok(Self {
            config: cfg,
            device: candle::Device::Cpu,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn decode(&self, codes: &candle::Tensor) -> candle::Result<candle::Tensor> {
        // Stub implementation - returns a zero tensor of appropriate shape
        let shape = codes.shape();
        let output_shape = if shape.dims().len() >= 2 {
            let mut new_shape = shape.dims().to_vec();
            new_shape[new_shape.len() - 1] = (new_shape[new_shape.len() - 1] * 4).max(1024); // Approximate audio expansion
            new_shape
        } else {
            vec![1024] // Default audio length
        };
        candle::Tensor::zeros(output_shape, candle::DType::F32, &self.device)
    }

    pub fn encode(&self, _xs: &candle::Tensor) -> candle::Result<candle::Tensor> {
        // Stub implementation - returns a zero tensor
        candle::Tensor::zeros((1, 16, 100), candle::DType::U32, &self.device)
    }

    pub fn reset_state(&self) {
        // Stub implementation - nothing to reset in minimal version
    }
}

pub fn load(_model_file: &str, _num_codebooks: Option<usize>, dev: &candle::Device) -> candle::Result<Mimi> {
    let cfg = Config {
        channels: 1,
        sample_rate: 24_000.0,
        frame_rate: 12.5,
        renormalize: true,
        resample_method: ResampleMethod::Conv,
        transformer: transformer::Config::default(),
        quantizer_n_q: _num_codebooks.unwrap_or(16),
        quantizer_bins: 2048,
        quantizer_dim: 256,
    };

    let mut mimi = Mimi::new(cfg, candle_nn::VarBuilder::zeros(candle::DType::F32, dev))?;
    mimi.device = dev.clone();
    Ok(mimi)
}
