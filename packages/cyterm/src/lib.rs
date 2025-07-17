// Removed custom acceleration constraint - following candle's CPU-first philosophy
// CPU-only builds are fully supported and recommended for cross-platform compatibility

#[cfg(all(
    not(feature = "microphone"),
    not(feature = "encodec"),
    not(feature = "mimi"),
    not(feature = "snac")
))]
compile_error!("At least one audio feature must be enabled: microphone, encodec, mimi, or snac");

pub mod asr;
pub mod colors;
pub mod features;
pub mod llm;
pub mod pixmap;
pub mod soft_backend;
pub mod ui;
pub mod wake_word; // KWS-based keyword detector // Whisper microphone pipeline (main entry)

pub use pixmap::RgbPixmap;
pub use soft_backend::SoftBackend;

// Re-export the main API that main.rs expects
pub use wake_word::WakeWordDetector;

pub use features::*;
