/// Wakeword comparison and scoring utilities.
pub mod comp;
/// Neural network models for wakeword detection.
pub mod nn;
mod wakeword_detector;
mod wakeword_file;
/// Wakeword model types and operations.
pub mod wakeword_model;
mod wakeword_ref;

pub use comp::{WakewordRefBuildFromBuffers, WakewordRefBuildFromFiles};
pub(crate) use wakeword_detector::WakewordDetector;
pub(crate) use wakeword_file::WakewordFile;
pub use wakeword_file::{WakewordLoad, WakewordSave};
pub use wakeword_model::{ModelType, ModelWeights, TensorData, WakewordModel};
pub use wakeword_ref::WakewordRef;
