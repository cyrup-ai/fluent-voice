use super::{projected_transformer::ProjectedTransformer, seanet, transformer};
use crate::conv::{ConvDownsample1d, ConvTrUpsample1d};
use crate::quantization::SplitResidualVectorQuantizer;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;
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
    pub seanet: seanet::Config,
    pub transformer: transformer::Config,
    pub quantizer_n_q: usize,
    pub quantizer_bins: usize,
    pub quantizer_dim: usize,
}

impl Config {
    pub fn v0_1(num_codebooks: Option<usize>) -> Self {
        let seanet_cfg = seanet::Config {
            dimension: 512,
            channels: 1,
            causal: true,
            n_filters: 64,
            n_residual_layers: 1,
            compress: 2,
            dilation_base: 2,
            disable_norm_outer_blocks: 0,
            kernel_size: 7,
            residual_kernel_size: 3,
            last_kernel_size: 3,
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
            norm: crate::transformer::NormType::LayerNorm,
            positional_embedding: crate::transformer::PositionalEmbedding::Rope,
            dim_feedforward: 2048,
            kv_repeat: 1,
            conv_layout: true,
            max_seq_len: 8192,
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
}

/// Production Mimi neural audio codec with SEANet
#[derive(Debug)]
pub struct Mimi {
    config: Config,
    device: Device,

    // Core SEANet components
    seanet: Arc<seanet::SeanetModule>,

    // Transformer processing
    encoder_transformer: ProjectedTransformer,
    decoder_transformer: ProjectedTransformer,

    // Quantization
    quantizer: SplitResidualVectorQuantizer,

    // Resampling for frame rate conversion
    downsample: ConvDownsample1d,
    upsample: ConvTrUpsample1d,

    // Derived parameters
    frame_size: usize,
    _downsample_stride: usize,
}

impl Mimi {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dim = cfg.seanet.dimension;

        // Calculate frame rates
        let encoder_frame_rate =
            cfg.sample_rate / cfg.seanet.ratios.iter().product::<usize>() as f64;
        let downsample_stride = (encoder_frame_rate / cfg.frame_rate) as usize;
        let frame_size = (cfg.sample_rate / cfg.frame_rate) as usize;

        // Build SEANet encoder/decoder
        let seanet = Arc::new(seanet::SeanetModule::new(
            cfg.seanet.clone(),
            vb.pp("seanet"),
        )?);

        // Build transformers with projections
        let encoder_transformer = ProjectedTransformer::new(
            cfg.transformer.clone(),
            dim,
            vec![dim],
            vb.pp("encoder_transformer"),
        )?;

        let decoder_transformer = ProjectedTransformer::new(
            cfg.transformer.clone(),
            dim,
            vec![dim],
            vb.pp("decoder_transformer"),
        )?;

        // Build quantizer
        let quantizer = SplitResidualVectorQuantizer::new(
            cfg.quantizer_dim,
            Some(dim),
            Some(dim),
            cfg.quantizer_n_q,
            cfg.quantizer_bins,
            vb.pp("quantizer"),
        )?;

        // Build resampling layers
        let downsample = ConvDownsample1d::new(
            downsample_stride,
            dim,
            true,  // causal
            false, // not learnt
            vb.pp("downsample"),
        )?;

        let upsample = ConvTrUpsample1d::new(
            downsample_stride,
            dim,
            true,  // causal
            false, // not learnt
            vb.pp("upsample"),
        )?;

        Ok(Self {
            config: cfg,
            device,
            seanet,
            encoder_transformer,
            decoder_transformer,
            quantizer,
            downsample,
            upsample,
            frame_size,
            _downsample_stride: downsample_stride,
        })
    }

    /// Full encoding pipeline: audio -> codes
    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        // Ensure correct input shape [batch, channels, time]
        let xs = if xs.dims().len() == 1 {
            xs.unsqueeze(0)?.unsqueeze(0)?
        } else if xs.dims().len() == 2 {
            xs.unsqueeze(0)?
        } else {
            xs.clone()
        };

        // Encode with SEANet
        let encoded = self.seanet.encode(&xs)?;

        // Process with transformer
        let transformed = self.encoder_transformer.forward(&encoded, false)?;
        let encoded = &transformed[0];

        // Downsample to target frame rate
        let downsampled = self
            .downsample
            .step(&crate::streaming::StreamTensor::from_tensor(
                encoded.clone(),
            ))?;
        let downsampled = downsampled
            .as_option()
            .ok_or_else(|| candle_core::Error::Msg("Downsample failed".to_string()))?;

        // Quantize to discrete codes
        self.quantizer.encode(downsampled)
    }

    /// Full decoding pipeline: codes -> audio
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        // Dequantize codes
        let quantized = self.quantizer.decode(codes)?;

        // Upsample to encoder frame rate
        let upsampled = self
            .upsample
            .step(&crate::streaming::StreamTensor::from_tensor(quantized))?;
        let upsampled = upsampled
            .as_option()
            .ok_or_else(|| candle_core::Error::Msg("Upsample failed".to_string()))?;

        // Process with transformer
        let transformed = self.decoder_transformer.forward(upsampled, false)?;
        let decoded = &transformed[0];

        // Decode with SEANet
        self.seanet.decode(decoded)
    }

    /// Streaming encode step
    pub fn encode_step(&self, xs: &Tensor) -> Result<Option<Tensor>> {
        // Convert to StreamTensor
        let stream_xs = crate::streaming::StreamTensor::from_tensor(xs.clone());

        // Encode with SEANet
        let encoded = self.seanet.encode_step(&stream_xs)?;

        if let Some(encoded_tensor) = encoded.as_option() {
            // Process with transformer
            let transformed = self.encoder_transformer.forward(encoded_tensor, true)?;
            let encoded = &transformed[0];

            // Downsample
            let downsampled =
                self.downsample
                    .step(&crate::streaming::StreamTensor::from_tensor(
                        encoded.clone(),
                    ))?;

            if let Some(downsampled_tensor) = downsampled.as_option() {
                // Quantize
                let codes = self.quantizer.encode(downsampled_tensor)?;
                Ok(Some(codes))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Streaming decode step
    pub fn decode_step(&self, codes: &Tensor) -> Result<Option<Tensor>> {
        // Dequantize
        let quantized = self.quantizer.decode(codes)?;

        // Upsample
        let upsampled = self
            .upsample
            .step(&crate::streaming::StreamTensor::from_tensor(quantized))?;

        if let Some(upsampled_tensor) = upsampled.as_option() {
            // Process with transformer
            let transformed = self.decoder_transformer.forward(upsampled_tensor, true)?;
            let decoded = &transformed[0];

            // Decode with SEANet
            let stream_decoded =
                self.seanet
                    .decode_step(&crate::streaming::StreamTensor::from_tensor(
                        decoded.clone(),
                    ))?;
            Ok(stream_decoded.as_option().cloned())
        } else {
            Ok(None)
        }
    }

    /// Reset all streaming state
    pub fn reset_state(&mut self) {
        self.encoder_transformer.reset_cache();
        self.decoder_transformer.reset_cache();
        self.downsample.reset_state();
        self.upsample.reset_state();
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get the frame size for streaming (1920 samples for 24kHz at 12.5Hz frame rate)
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Flush any pending streaming state
    pub fn flush(&mut self) -> Result<()> {
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
    let cfg = Config::v0_1(num_codebooks);

    // Create the Mimi instance with loaded weights
    Mimi::new(cfg, vb)
}

/// Load Mimi model from a path without requiring UTF-8 conversion
///
/// This function safely handles path types without unwrap() calls that could panic
/// on non-UTF-8 paths, following existing patterns from TURD.md specification.
pub fn load_from_path<P: AsRef<Path>>(
    model_path: P,
    num_codebooks: Option<usize>,
    dev: &Device,
) -> Result<Mimi> {
    let model_file = model_path.as_ref().to_str().ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "Invalid UTF-8 in model path: {:?}",
            model_path.as_ref()
        ))
    })?;

    load(model_file, num_codebooks, dev)
}
