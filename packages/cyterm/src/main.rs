//! Cyterm: A real-time, voice-enabled terminal emulator powered by fluent-voice.

use anyhow::Result;
use fluent_voice::prelude::*;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    init_logging();

    println!("Starting real-time STT with fluent-voice...");

    println!("\nListening for your voice... (Ctrl-C to quit)");

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
        .on_chunk(|result| match result {
            Ok(segment) => segment,
            Err(e) => {
                eprintln!("Recognition error: {}", e);
                TranscriptionSegmentImpl::new(format!("[ERROR] {}", e), 0, 0, None)
            }
        })
        .listen(|result| match result {
            Ok(conversation) => {
                use futures_util::StreamExt;
                Box::pin(conversation.into_stream().map(|item| match item {
                    Ok(kyutai_segment) => {
                        // Convert KyutaiTranscriptSegment to TranscriptionSegmentImpl
                        Ok(fluent_voice::TranscriptionSegmentImpl::new(
                            kyutai_segment.text().to_string(),
                            kyutai_segment.start_ms(),
                            kyutai_segment.end_ms(),
                            None,
                        ))
                    }
                    Err(e) => Err(e),
                }))
            }
            Err(e) => panic!("Failed to create STT conversation: {}", e),
        });

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
