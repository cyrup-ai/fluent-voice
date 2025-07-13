mod wakeword_model_train;
pub mod wakeword_nn;

pub use wakeword_model_train::{WakewordModelTrainOptions, train as WakewordModelTrain};
pub(crate) use wakeword_nn::WakewordNN;
