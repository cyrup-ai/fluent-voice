/// Display module for visualization components
pub mod oscillator;
pub mod spectroscope;
pub mod vector;

// Re-export common types from oscillator for convenience
pub use crate::oscillator::{Dimension, DisplayMode, GraphConfig, update_value_f, update_value_i};
