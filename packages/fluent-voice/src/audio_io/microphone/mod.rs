pub mod audio_processing;
pub mod cli;
pub mod decoder;
pub mod device;
pub mod model;
pub mod stream;

// Re-export main record function for backward compatibility
pub use audio_processing::record;

// Re-export public types for external usage
pub use cli::{Args, Task, WhichModel};
pub use decoder::Decoder;
pub use model::Model;
pub use stream::AudioStream;
