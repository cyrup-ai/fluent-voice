//! Cyterm: A real-time, voice-enabled terminal emulator powered by fluent-voice.

use anyhow::Result;
use fluent_voice::prelude::*;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    init_logging();

    println!("Starting real-time STT with fluent-voice...");

    let mut transcript_stream = FluentVoice::stt()
        .conversation()
        .on_prediction(|transcription, prediction| {
            // In a real terminal, you would use termcolor here to draw the fading animation.
            // For this integration, we just print the prediction.
            print!("\r\x1B[K"); // Clear line and move cursor to start
            print!("{} [prediction: {}]", transcription, prediction);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        })
        .with_microphone()
        .listen(|conv_result| match conv_result {
            Ok(conv) => conv.into_stream(),
            Err(e) => {
                eprintln!("Error creating conversation: {}", e);
                // Return an empty stream on error
                futures_util::stream::empty().boxed()
            }
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
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
}
