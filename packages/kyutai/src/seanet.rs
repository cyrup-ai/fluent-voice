//! SEANet neural audio encoder-decoder implementation
//!
//! Based on Kyutai/Moshi SEANet architecture for streaming neural audio compression.

use crate::streaming::StreamTensor;
use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, ConvTranspose1d, Module, VarBuilder};

/// SEANet configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub channels: usize,
    pub dimension: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub causal: bool,
    pub compress: usize,
    pub dilation_base: usize,
    pub disable_norm_outer_blocks: usize,
    pub true_skip: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            channels: 1,
            dimension: 512,
            n_filters: 64,
            n_residual_layers: 1,
            ratios: vec![8, 6, 5, 4],
            kernel_size: 7,
            residual_kernel_size: 3,
            last_kernel_size: 3,
            causal: true,
            compress: 2,
            dilation_base: 2,
            disable_norm_outer_blocks: 0,
            true_skip: true,
        }
    }
}

/// SEANet residual block
#[derive(Debug)]
pub struct SEANetResnetBlock {
    block: Vec<Conv1d>,
    shortcut: Option<Conv1d>,
    _compress: usize,
}

impl SEANetResnetBlock {
    pub fn new(
        dim: usize,
        kernel_sizes: &[usize],
        dilations: &[usize],
        config: &Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden = dim / config.compress;
        let mut block = Vec::new();

        for (i, (&kernel_size, &dilation)) in kernel_sizes.iter().zip(dilations).enumerate() {
            let in_channels = if i == 0 { dim } else { hidden };
            let out_channels = if i == kernel_sizes.len() - 1 {
                dim
            } else {
                hidden
            };

            let conv_config = candle_nn::Conv1dConfig {
                dilation,
                padding: if config.causal {
                    0
                } else {
                    (kernel_size - 1) / 2
                },
                ..Default::default()
            };
            let conv = candle_nn::conv1d(
                in_channels,
                out_channels,
                kernel_size,
                conv_config,
                vb.pp(&format!("block.{}", i)),
            )?;
            block.push(conv);
        }

        let shortcut = if config.true_skip {
            None
        } else {
            let conv = candle_nn::conv1d(dim, dim, 1, Default::default(), vb.pp("shortcut"))?;
            Some(conv)
        };

        Ok(Self {
            block,
            shortcut,
            _compress: config.compress,
        })
    }

    pub fn step(&self, xs: &StreamTensor) -> Result<StreamTensor> {
        if let Some(xs_tensor) = xs.as_option() {
            let result = self.forward(xs_tensor)?;
            Ok(StreamTensor::from_tensor(result))
        } else {
            Ok(StreamTensor::empty())
        }
    }
}

impl Module for SEANetResnetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();

        // Apply residual block layers with ELU activation between layers
        for (i, conv) in self.block.iter().enumerate() {
            if i > 0 {
                // Apply ELU activation before each conv layer (except first)
                x = x.elu(1.0)?;
            }
            x = conv.forward(&x)?;
        }

        // Apply shortcut connection
        let shortcut_out = if let Some(ref shortcut) = self.shortcut {
            shortcut.forward(xs)?
        } else {
            xs.clone()
        };

        x.add(&shortcut_out)
    }
}

/// SEANet encoder
#[derive(Debug)]
pub struct SEANetEncoder {
    initial_conv: Conv1d,
    encoder_blocks: Vec<EncoderBlock>,
    final_conv: Conv1d,
    _config: Config,
}

#[derive(Debug)]
struct EncoderBlock {
    resnet_blocks: Vec<SEANetResnetBlock>,
    downsample: Conv1d,
}

impl SEANetEncoder {
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let mut mult = 1;
        let reversed_ratios: Vec<usize> = config.ratios.iter().rev().cloned().collect();

        // Initial convolution
        let initial_conv = candle_nn::conv1d(
            config.channels,
            config.n_filters,
            config.kernel_size,
            Default::default(),
            vb.pp("initial"),
        )?;

        // Encoder blocks with downsampling
        let mut encoder_blocks = Vec::new();
        for (i, &ratio) in reversed_ratios.iter().enumerate() {
            // Create residual blocks
            let mut resnet_blocks = Vec::new();
            for j in 0..config.n_residual_layers {
                let resnet_block = SEANetResnetBlock::new(
                    mult * config.n_filters,
                    &[config.residual_kernel_size, 1],
                    &[config.dilation_base.pow(j as u32) as usize, 1],
                    &config,
                    vb.pp(&format!("encoder.{}.resnet.{}", i, j)),
                )?;
                resnet_blocks.push(resnet_block);
            }

            // Downsampling convolution
            let downsample_config = candle_nn::Conv1dConfig {
                stride: ratio,
                padding: if config.causal { 0 } else { ratio - 1 },
                ..Default::default()
            };
            let downsample = candle_nn::conv1d(
                mult * config.n_filters,
                mult * config.n_filters * 2,
                ratio * 2,
                downsample_config,
                vb.pp(&format!("encoder.{}.downsample", i)),
            )?;

            encoder_blocks.push(EncoderBlock {
                resnet_blocks,
                downsample,
            });

            mult *= 2;
        }

        // Final convolution to latent dimension
        let final_conv = candle_nn::conv1d(
            mult * config.n_filters,
            config.dimension,
            config.last_kernel_size,
            Default::default(),
            vb.pp("final"),
        )?;

        Ok(Self {
            initial_conv,
            encoder_blocks,
            final_conv,
            _config: config,
        })
    }

    pub fn encode_step(&self, input: &StreamTensor) -> Result<StreamTensor> {
        if let Some(input_tensor) = input.as_option() {
            let result = self.forward(input_tensor)?;
            Ok(StreamTensor::from_tensor(result))
        } else {
            Ok(StreamTensor::empty())
        }
    }
}

impl Module for SEANetEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = self.initial_conv.forward(xs)?;

        // Process through encoder blocks
        for block in &self.encoder_blocks {
            // Apply residual layers
            for resnet_block in &block.resnet_blocks {
                x = resnet_block.forward(&x)?;
            }

            // Apply ELU activation before downsampling
            x = x.elu(1.0)?;

            // Apply downsampling
            x = block.downsample.forward(&x)?;
        }

        // Apply final ELU activation and final convolution
        x = x.elu(1.0)?;
        self.final_conv.forward(&x)
    }
}

/// SEANet decoder
#[derive(Debug)]
pub struct SEANetDecoder {
    initial_conv: Conv1d,
    decoder_blocks: Vec<DecoderBlock>,
    final_conv: Conv1d,
    _config: Config,
}

#[derive(Debug)]
struct DecoderBlock {
    upsample: ConvTranspose1d,
    resnet_blocks: Vec<SEANetResnetBlock>,
}

impl SEANetDecoder {
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let mut mult = 2_usize.pow(config.ratios.len() as u32);

        // Initial convolution from latent dimension
        let initial_conv = candle_nn::conv1d(
            config.dimension,
            mult * config.n_filters,
            config.kernel_size,
            Default::default(),
            vb.pp("initial"),
        )?;

        // Decoder blocks with upsampling
        let mut decoder_blocks = Vec::new();
        for (i, &ratio) in config.ratios.iter().enumerate() {
            // Upsampling transpose convolution
            let upsample_config = candle_nn::ConvTranspose1dConfig {
                stride: ratio,
                padding: if config.causal { 0 } else { ratio - 1 },
                ..Default::default()
            };
            let upsample = candle_nn::conv_transpose1d(
                mult * config.n_filters,
                mult * config.n_filters / 2,
                ratio * 2,
                upsample_config,
                vb.pp(&format!("decoder.{}.upsample", i)),
            )?;

            // Create residual blocks
            let mut resnet_blocks = Vec::new();
            for j in 0..config.n_residual_layers {
                let resnet_block = SEANetResnetBlock::new(
                    mult * config.n_filters / 2,
                    &[config.residual_kernel_size, 1],
                    &[config.dilation_base.pow(j as u32) as usize, 1],
                    &config,
                    vb.pp(&format!("decoder.{}.resnet.{}", i, j)),
                )?;
                resnet_blocks.push(resnet_block);
            }

            decoder_blocks.push(DecoderBlock {
                upsample,
                resnet_blocks,
            });

            mult /= 2;
        }

        // Final convolution to audio channels
        let final_conv = candle_nn::conv1d(
            config.n_filters,
            config.channels,
            config.last_kernel_size,
            Default::default(),
            vb.pp("final"),
        )?;

        Ok(Self {
            initial_conv,
            decoder_blocks,
            final_conv,
            _config: config,
        })
    }

    pub fn decode_step(&self, latent: &StreamTensor) -> Result<StreamTensor> {
        if let Some(latent_tensor) = latent.as_option() {
            let result = self.forward(latent_tensor)?;
            Ok(StreamTensor::from_tensor(result))
        } else {
            Ok(StreamTensor::empty())
        }
    }
}

impl Module for SEANetDecoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = self.initial_conv.forward(xs)?;

        // Process through decoder blocks
        for block in &self.decoder_blocks {
            // Apply ELU activation before upsampling
            x = x.elu(1.0)?;

            // Apply upsampling
            x = block.upsample.forward(&x)?;

            // Apply residual layers
            for resnet_block in &block.resnet_blocks {
                x = resnet_block.forward(&x)?;
            }
        }

        // Apply final ELU activation and final convolution
        x = x.elu(1.0)?;
        self.final_conv.forward(&x)
    }
}

/// Complete SEANet module
#[derive(Debug)]
pub struct SeanetModule {
    encoder: SEANetEncoder,
    decoder: SEANetDecoder,
    _config: Config,
}

impl SeanetModule {
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let encoder = SEANetEncoder::new(config.clone(), vb.pp("encoder"))?;
        let decoder = SEANetDecoder::new(config.clone(), vb.pp("decoder"))?;

        Ok(Self {
            encoder,
            decoder,
            _config: config,
        })
    }

    pub fn encode(&self, input: &Tensor) -> Result<Tensor> {
        self.encoder.forward(input)
    }

    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        self.decoder.forward(latent)
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let encoded = self.encode(input)?;
        self.decode(&encoded)
    }

    pub fn encode_step(&self, input: &StreamTensor) -> Result<StreamTensor> {
        self.encoder.encode_step(input)
    }

    pub fn decode_step(&self, encoded: &StreamTensor) -> Result<StreamTensor> {
        self.decoder.decode_step(encoded)
    }

    pub fn config(&self) -> &Config {
        &self._config
    }
}
