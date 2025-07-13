#[cfg(all(
    not(feature = "microphone"),
    not(feature = "encodec"),
    not(feature = "mimi"),
    not(feature = "snac")
))]
compile_error!("At least one audio feature must be enabled: microphone, encodec, mimi, or snac");

pub mod app;
pub mod audioio;
pub mod cfg;
pub mod display;
pub mod input;
pub mod music;
pub mod oscillator;
