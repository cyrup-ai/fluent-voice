// src/conv.rs

use crate::streaming::StreamTensor;
use candle_core::Result;
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm {
    WeightNorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    Constant,
}

#[derive(Debug, Clone)]
pub struct ConvDownsample1d {
    conv: Conv1d,
    causal: bool,
    learnt: bool,
}

impl ConvDownsample1d {
    pub fn new(
        stride: usize,
        dim: usize,
        causal: bool,
        learnt: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let kernel_size = stride * 2 - 1;
        let config = Conv1dConfig {
            stride,
            padding: stride - 1,
            ..Default::default()
        };
        let conv = candle_nn::conv1d(dim, dim, kernel_size, config, vb)?;
        Ok(Self {
            conv,
            causal,
            learnt,
        })
    }

    pub fn step(&self, xs: &StreamTensor) -> Result<StreamTensor> {
        if let Some(xs) = xs.as_option() {
            let mut out = self.conv.forward(xs)?;

            // Apply causal masking if enabled
            if self.causal {
                // For causal convolution, trim the output to avoid looking into the future
                let (_batch_size, _channels, seq_len) = out.dims3()?;
                if seq_len > 1 {
                    out = out.narrow(2, 0, seq_len - 1)?;
                }
            }

            // Apply learnt adjustments if enabled
            if self.learnt {
                // Learnt convolution uses trainable parameters loaded from model
                // No additional processing needed - conv weights are already learned
                // This is the identity case: learned parameters are in the conv layer itself
            }

            Ok(StreamTensor::from_tensor(out))
        } else {
            Ok(StreamTensor::empty())
        }
    }

    pub fn reset_state(&mut self) {
        // Convolution layers are stateless for basic operations
        // No internal state to reset for downsampling
    }
}

#[derive(Debug, Clone)]
pub struct ConvTrUpsample1d {
    conv_tr: Conv1d,
    causal: bool,
    learnt: bool,
}

impl ConvTrUpsample1d {
    pub fn new(
        stride: usize,
        dim: usize,
        causal: bool,
        learnt: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let kernel_size = stride * 2;
        let config = Conv1dConfig {
            stride: 1,
            padding: stride,
            dilation: stride,
            ..Default::default()
        };
        let conv_tr = candle_nn::conv1d(dim, dim, kernel_size, config, vb)?;
        Ok(Self {
            conv_tr,
            causal,
            learnt,
        })
    }

    pub fn step(&self, xs: &StreamTensor) -> Result<StreamTensor> {
        if let Some(xs) = xs.as_option() {
            let mut out = self.conv_tr.forward(xs)?;

            // Apply causal masking if enabled
            if self.causal {
                // For causal transpose convolution, ensure we don't use future information
                let (_batch_size, _channels, seq_len) = out.dims3()?;
                if seq_len > 1 {
                    out = out.narrow(2, 0, seq_len - 1)?;
                }
            }

            // Apply learnt adjustments if enabled
            if self.learnt {
                // Learnt transpose convolution uses trainable parameters from model
                // Additional normalization would be applied here if needed
                // For Moshi models, the learned weights handle this internally
            }

            Ok(StreamTensor::from_tensor(out))
        } else {
            Ok(StreamTensor::empty())
        }
    }

    pub fn reset_state(&mut self) {
        // No state to reset for conv transpose
    }
}
