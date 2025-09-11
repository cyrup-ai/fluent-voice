use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

// ------------ Data -----------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Max text sequence length (≥1, multiple of 128).
    pub text_length: usize,
    /// Max audio token length (≥1, multiple of 128).
    pub audio_length: usize,
    /// Number of EnCodec/VQ channels (default 9).
    pub channels: usize,
    pub text_pad_value: u32,
    pub audio_eos_value: u32,
    pub audio_pad_value: u32,
    pub audio_bos_value: u32,
    pub delay_pattern: Vec<u32>,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            text_length: 1024, // will be rounded up on validate()
            audio_length: 3072,
            channels: 9,
            text_pad_value: 0,
            audio_eos_value: 1024,
            audio_pad_value: 1025,
            audio_bos_value: 1026,
            delay_pattern: vec![0, 8, 9, 10, 11, 12, 13, 14, 15],
        }
    }
}

impl DataConfig {
    pub fn validate(&mut self) {
        let round128 = |x: usize| x.div_ceil(128) * 128;
        self.text_length = round128(self.text_length.max(1));
        self.audio_length = round128(self.audio_length.max(1));
        assert!(self.channels > 0, "channels must be > 0");
    }
}

// ------------ Encoder ---------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_hidden: usize,
    pub n_head: usize,
    pub head_dim: usize,
}

// ------------ Decoder ---------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_hidden: usize,
    pub gqa_query_heads: usize,
    pub kv_heads: usize,
    pub gqa_head_dim: usize,
    pub cross_query_heads: usize,
    pub cross_head_dim: usize,
}

// ------------ Model -----------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub encoder: EncoderConfig,
    pub decoder: DecoderConfig,
    pub src_vocab_size: usize,
    pub tgt_vocab_size: usize,
    pub dropout: f32,
    pub normalization_layer_epsilon: f32,
    pub weight_dtype: String,
    pub rope_min_timescale: u32,
    pub rope_max_timescale: u32,
}

// ------------ Training (placeholder) ------------
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingConfig {}

// ------------ DiaConfig (root) -------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiaConfig {
    pub version: String,
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub data: DataConfig,
}

impl Default for DiaConfig {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            model: ModelConfig {
                encoder: EncoderConfig {
                    n_layer: 12,
                    n_embd: 768,
                    n_hidden: 3072,
                    n_head: 12,
                    head_dim: 64,
                },
                decoder: DecoderConfig {
                    n_layer: 24,
                    n_embd: 1024,
                    n_hidden: 4096,
                    gqa_query_heads: 16,
                    kv_heads: 16,
                    gqa_head_dim: 64,
                    cross_query_heads: 16,
                    cross_head_dim: 64,
                },
                src_vocab_size: 32000,
                tgt_vocab_size: 1280,
                dropout: 0.1,
                normalization_layer_epsilon: 1e-5,
                weight_dtype: "float32".to_string(),
                rope_min_timescale: 1,
                rope_max_timescale: 10000,
            },
            training: TrainingConfig::default(),
            data: DataConfig::default(),
        }
    }
}

impl DiaConfig {
    /// Load a JSON config from disk.
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let txt = fs::read_to_string(path)?;
        let mut cfg: DiaConfig = serde_json::from_str(&txt)?;
        cfg.data.validate();
        Ok(cfg)
    }

    /// Save to disk (pretty‑printed).
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        if let Some(parent) = path.as_ref().parent()
            && !parent.exists()
        {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
    }
}
