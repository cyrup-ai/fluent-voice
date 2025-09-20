//! Base language model configuration types and core implementations

use super::basic_types::{ConditionerConfig, ConditionersConfig, LutConfig, TensorConfig};
use super::conditioners::{DepFormerConfig, FuserConfig};
use crate::transformer::{
    Config as TransformerConfig, CrossAttentionGating, NormType, PositionalEmbedding,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Language model configuration containing transformer and conditioning settings
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LmConfig {
    pub transformer: TransformerConfig,
    pub depformer: Option<DepFormerConfig>,
    pub fuser: Option<FuserConfig>,
    pub conditioners: Option<ConditionersConfig>,
    pub audio_vocab_size: usize,
    pub text_in_vocab_size: usize,
    pub text_out_vocab_size: usize,
    pub audio_codebooks: usize,
}

impl LmConfig {
    /// TTS model configuration for 1.6B parameter model (English/French)
    pub fn tts_1_6b_en_fr() -> Self {
        let lm_cfg = TransformerConfig {
            d_model: 2048,
            num_heads: 16,
            num_layers: 16,
            dim_feedforward: 8448,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 500,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: Some((CrossAttentionGating::Normal, NormType::RmsNorm, None)),
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        let mut conditioners = HashMap::new();
        conditioners.insert(
            "speaker_wavs".to_string(),
            ConditionerConfig::Tensor(TensorConfig { dim: 512 }),
        );
        conditioners.insert(
            "cfg".to_string(),
            ConditionerConfig::Lut(LutConfig {
                n_bins: 7,
                dim: 16,
                tokenizer: "noop".to_string(),
                possible_values: vec![
                    "1.0".into(),
                    "1.5".into(),
                    "2.0".into(),
                    "2.5".into(),
                    "3.0".into(),
                    "3.5".into(),
                    "4.0".into(),
                ],
            }),
        );
        conditioners.insert(
            "control".to_string(),
            ConditionerConfig::Lut(LutConfig {
                n_bins: 1,
                dim: 2048,
                tokenizer: "noop".to_string(),
                possible_values: vec!["ok".into()],
            }),
        );
        Self {
            transformer: lm_cfg,
            depformer: Some(DepFormerConfig {
                transformer: TransformerConfig {
                    d_model: 1024,
                    num_heads: 16,
                    num_layers: 4,
                    dim_feedforward: 3072,
                    causal: true,
                    norm_first: true,
                    bias_ff: false,
                    bias_attn: false,
                    layer_scale: None,
                    context: 32,
                    max_period: 10000,
                    use_conv_block: false,
                    use_conv_bias: true,
                    cross_attention: None,
                    gating: Some(candle_nn::Activation::Silu),
                    norm: NormType::RmsNorm,
                    positional_embedding: PositionalEmbedding::None,
                    conv_layout: false,
                    conv_kernel_size: 3,
                    kv_repeat: 1,
                    max_seq_len: 4096,
                    shared_cross_attn: false,
                },
                num_slices: 32,
                low_rank_embeddings: Some(128),
                shared: true,
                multi_linear: true,
                weights_per_step: true,
                pos_emb: "none".to_string(),
                weights_per_step_schedule: Some(vec![
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10,
                    10, 10, 10, 10, 10, 10,
                ]),
            }),
            fuser: Some(FuserConfig {
                cross_attention_pos_emb: true,
                cross_attention_pos_emb_scale: 1.0,
                sum: vec!["control".into(), "cfg".into()],
                prepend: vec![],
                cross: vec!["speaker_wavs".into()],
            }),
            conditioners: Some(conditioners),
            audio_vocab_size: 2049,
            text_in_vocab_size: 8001,
            text_out_vocab_size: 8000,
            audio_codebooks: 32,
        }
    }

    /// STT model configuration for 2.6B parameter model (English)
    pub fn stt_2_6b_en() -> Self {
        let lm_cfg = TransformerConfig {
            d_model: 2048,
            num_heads: 32,
            num_layers: 48,
            dim_feedforward: 8448,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 375,
            max_period: 100000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        Self {
            transformer: lm_cfg,
            depformer: Some(DepFormerConfig {
                transformer: TransformerConfig {
                    d_model: 1024,
                    num_heads: 16,
                    num_layers: 6,
                    dim_feedforward: 4096,
                    causal: true,
                    norm_first: true,
                    bias_ff: false,
                    bias_attn: false,
                    layer_scale: None,
                    context: 0,
                    max_period: 10000,
                    use_conv_block: false,
                    use_conv_bias: true,
                    cross_attention: None,
                    gating: Some(candle_nn::Activation::Silu),
                    norm: NormType::RmsNorm,
                    positional_embedding: PositionalEmbedding::None,
                    conv_layout: false,
                    conv_kernel_size: 3,
                    kv_repeat: 1,
                    max_seq_len: 4096,
                    shared_cross_attn: false,
                },
                num_slices: 0,
                low_rank_embeddings: None,
                shared: true,
                multi_linear: true,
                weights_per_step: true,
                pos_emb: "none".to_string(),
                weights_per_step_schedule: None,
            }),
            fuser: None,
            conditioners: None,
            audio_vocab_size: 2049,
            text_in_vocab_size: 4001,
            text_out_vocab_size: 4000,
            audio_codebooks: 32,
        }
    }
}
