// Whisper CLI - Live Microphone Transcription using WhisperSttBuilder API

use anyhow::Result;
use fluent_voice_whisper::{AudioFormat, MicBackend, SpeechSource, WhisperSttBuilder};
use termcolor::{ColoredMessage, ThemeConfig, error_x, info_i, set_global_theme};

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[tokio::main]
async fn main() -> Result<()> {
    // Set Cyrup.ai theme
    set_global_theme(ThemeConfig::Default);

    // Display header
    if let Err(e) = ColoredMessage::cyrup_header().println() {
        eprintln!("Failed to print header: {}", e);
    }
    info_i!("Starting live microphone transcription...");
    info_i!("Press Ctrl+C to stop");

    let _result = WhisperSttBuilder::new()
        .with_source(SpeechSource::Microphone {
            backend: MicBackend::Default,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16000,
        })
        .on_chunk(|chunk_result| match chunk_result {
            Ok(chunk) => {
                if !chunk.text.trim().is_empty()
                    && let Err(e) = ColoredMessage::new().text_primary(&chunk.text).print()
                {
                    eprintln!("Failed to print transcription: {}", e);
                }
            }
            Err(e) => {
                error_x!("Transcription error: {}", e);
            }
        })
        .listen(|result| {
            match result {
                Ok(_) => {
                    if let Err(e) = ColoredMessage::new()
                        .newline()
                        .success("Transcription session ended")
                        .println()
                    {
                        eprintln!("Failed to print session end message: {}", e);
                    }
                }
                Err(ref e) => {
                    error_x!("Session error: {}", e);
                }
            }
            result
        })
        .await?;

    Ok(())
}
