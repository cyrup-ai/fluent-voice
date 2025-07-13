#[cfg(not(any(feature = "cuda", feature = "metal", feature = "accelerate", feature = "mkl")))]
compile_error!("At least one candle acceleration feature must be enabled: cuda, metal, accelerate, or mkl");

pub mod app;
pub mod audio;
pub mod audio_io;
pub mod codec;
pub mod config;
pub mod generation;
pub mod layers;
pub mod model;
pub mod optimizations;
pub mod setup;
pub mod state;
pub mod ui;
pub mod voice;

use setup::setup;
