// Whisper CLI - Live Microphone Transcription

use anyhow::Result;

#[cfg(feature = "microphone")]
use fluent_voice_whisper::microphone;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[tokio::main]
async fn main() -> Result<()> {
    #[cfg(feature = "microphone")]
    return microphone::record().await;

    #[cfg(not(feature = "microphone"))]
    return Err(anyhow::anyhow!(
        "Microphone feature not enabled. Run with --features microphone"
    ));
}
