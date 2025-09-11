// src/audio.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}
