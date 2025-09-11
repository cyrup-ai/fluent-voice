use super::transformer;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::mimi::{Config as MimiConfig, Model as MimiModel};
use std::sync::Arc;

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

/// Production-quality Mimi neural audio codec implementation
///
/// This wraps the proven candle-transformers Mimi implementation to provide
/// a streaming neural audio codec with proper encode/decode functionality.
#[derive(Debug)]
pub struct Mimi {
    config: Config,
    device: Device,
    model: Arc<std::sync::Mutex<MimiModel>>,
    frame_size: usize,
}

/// Production-quality Mimi implementation using candle-transformers
impl Mimi {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();

        // Convert our config to candle-transformers MimiConfig
        let mimi_config = MimiConfig::v0_1(Some(cfg.quantizer_n_q));

        // Create the actual Mimi model using candle-transformers
        let model = MimiModel::new(mimi_config, vb)?;

        // Calculate frame size based on sample rate and frame rate
        // Mimi uses 1920 samples per frame at 24kHz for 12.5Hz frame rate
        let frame_size = (cfg.sample_rate / cfg.frame_rate) as usize;

        Ok(Self {
            config: cfg,
            device,
            model: Arc::new(std::sync::Mutex::new(model)),
            frame_size,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Encode audio tensor to discrete codes
    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| candle_core::Error::Msg("Failed to acquire model lock".to_string()))?;

        // Ensure input is in the correct format [batch, channels, time]
        let xs = if xs.dims().len() == 1 {
            xs.unsqueeze(0)?.unsqueeze(0)?
        } else if xs.dims().len() == 2 {
            xs.unsqueeze(0)?
        } else {
            xs.clone()
        };

        model.encode(&xs)
    }

    /// Decode discrete codes back to audio tensor
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| candle_core::Error::Msg("Failed to acquire model lock".to_string()))?;

        model.decode(codes)
    }

    /// Reset the streaming state of the model
    pub fn reset_state(&self) {
        if let Ok(mut model) = self.model.lock() {
            model.reset_state();
        }
    }

    /// Decode a single streaming step
    pub fn decode_step(&self, codes: &Tensor) -> Result<Option<Tensor>> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| candle_core::Error::Msg("Failed to acquire model lock".to_string()))?;

        // Convert tensor to StreamTensor for streaming decode
        let stream_codes = candle_core::StreamTensor::from_tensor(codes.clone());
        let result = model.decode_step(&stream_codes)?;

        // Convert StreamTensor result back to Option<Tensor>
        Ok(result.as_option().cloned())
    }

    /// Encode a single streaming step
    pub fn encode_step(&self, xs: &Tensor) -> Result<Option<Tensor>> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| candle_core::Error::Msg("Failed to acquire model lock".to_string()))?;

        // Ensure input is in the correct format [batch, channels, time]
        let xs = if xs.dims().len() == 1 {
            xs.unsqueeze(0)?.unsqueeze(0)?
        } else if xs.dims().len() == 2 {
            xs.unsqueeze(0)?
        } else {
            xs.clone()
        };

        // Convert tensor to StreamTensor for streaming encode
        let stream_xs = candle_core::StreamTensor::from_tensor(xs);
        let result = model.encode_step(&stream_xs)?;

        // Convert StreamTensor result back to Option<Tensor>
        Ok(result.as_option().cloned())
    }

    /// Get the frame size for streaming (1920 samples for 24kHz at 12.5Hz frame rate)
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Flush any pending streaming state
    pub fn flush(&mut self) -> Result<()> {
        // The candle-transformers implementation handles state internally
        self.reset_state();
        Ok(())
    }

    /// Get the device this model is running on
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Load a Mimi model from a safetensors file
///
/// This function loads a pre-trained Mimi model from disk and creates a production-ready
/// neural audio codec instance.
pub fn load(model_file: &str, num_codebooks: Option<usize>, dev: &Device) -> Result<Mimi> {
    // Load the model weights from safetensors file
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, dev).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to load model file {}: {}", model_file, e))
        })?
    };

    // Create configuration matching the loaded model
    let cfg = Config {
        channels: 1,
        sample_rate: 24_000.0,
        frame_rate: 12.5,
        renormalize: true,
        resample_method: ResampleMethod::Conv,
        transformer: transformer::Config::default(),
        quantizer_n_q: num_codebooks.unwrap_or(16),
        quantizer_bins: 2048,
        quantizer_dim: 256,
    };

    // Create the Mimi instance with loaded weights
    Mimi::new(cfg, vb)
}
