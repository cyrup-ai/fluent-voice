//! Neural network module for wake word detection.
//!
//! This module contains the neural network implementations and training
//! functionality for wake word models. It includes both the neural network
//! architecture and the training pipeline.

mod wakeword_model_train;
pub mod wakeword_nn;

pub use wakeword_model_train::{WakewordModelTrainOptions, train as WakewordModelTrain};
pub(crate) use wakeword_nn::WakewordNN;
