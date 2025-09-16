//! Cyterm: A real-time, voice-enabled terminal emulator powered by fluent-voice.

use anyhow::Result;
use fluent_voice::prelude::*;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    init_logging();

    println!("Starting real-time STT with fluent-voice...");

    let mut transcript_stream = FluentVoice::stt()
        .with_source(SpeechSource::Microphone {
            backend: MicBackend::Default,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        })
        .vad_mode(VadMode::Accurate)
        .on_prediction(|transcription, prediction| {
            // In a real terminal, you would use termcolor here to draw the fading animation.
            // For this integration, we just print the prediction.
            print!("\r\x1B[K"); // Clear line and move cursor to start
            print!("{} [prediction: {}]", transcription, prediction);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        })
        .listen(|result| match result {
            Ok(conversation) => Ok(conversation.into_stream()),
            Err(e) => Err(e),
        })
        .await;

    println!("\nListening for your voice... (Ctrl-C to quit)");

    // Process the final transcript segments as they are confirmed.
    while let Some(result) = transcript_stream.next().await {
        match result {
            Ok(segment) => {
                // Clear the prediction line and print the final segment.
                print!("\r\x1B[K");
                println!("Segment: {}", segment.text());
            }
            Err(e) => {
                eprintln!("\nRecognition error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

fn init_logging() {
    env_logger::init();
}
