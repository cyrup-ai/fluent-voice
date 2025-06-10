pub mod asr;
pub mod colors;
pub mod features;
pub mod llm;
pub mod pixmap;
pub mod soft_backend;
pub mod wake_word; // KWS-based keyword detector // Whisper microphone pipeline (main entry)

pub mod llm;
pub mod ui;

pub use pixmap::RgbPixmap;
pub use soft_backend::SoftBackend;

// Re-export the main API that main.rs expects
pub use wake_word::WakeWordDetector;

pub use features::*;
