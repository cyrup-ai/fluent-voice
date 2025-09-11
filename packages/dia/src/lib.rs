// Removed custom acceleration constraint - following candle's CPU-first philosophy
// CPU-only builds are fully supported and recommended for cross-platform compatibility

// Re-export candle types for internal use
pub use candle_core::{DType, Device, Result as CandleResult, Tensor};
pub use candle_nn::{Module, VarBuilder, VarMap, ops};

// pub mod app;  // Temporarily disabled due to itertools conflict
pub mod audio;
pub mod audio_io;
pub mod codec;
pub mod config;
pub mod generation;
pub mod layers;
pub mod model;
pub mod model_downloader;
pub mod optimizations;
pub mod setup;
pub mod state;
pub mod ui;
pub mod voice;

use setup::setup;
