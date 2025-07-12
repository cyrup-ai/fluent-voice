//! Production STT Demo
//!
//! Demonstrates the complete production pipeline with:
//! - Wake word detection using Koffee ("syrup" model)
//! - Voice Activity Detection using VAD crate
//! - Speech-to-Text using Whisper (placeholder transcription)
//! - Real microphone capture and processing
//!
//! This example shows the DefaultSTTEngine in action with all integrated components.

use fluent_voice::prelude::*;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎙️  FluentVoice Production STT Demo");
    println!("=====================================");
    println!("Features:");
    println!("- Wake word detection: 'syrup' activation");
    println!("- Voice Activity Detection: Real-time VAD processing");
    println!("- Speech-to-Text: Whisper transcription pipeline");
    println!("- Microphone: Live audio capture and processing");
    println!();

    // Create the default STT engine with production pipeline
    println!("🔧 Initializing DefaultSTTEngine with production components...");
    let stt_engine = FluentVoice::stt();

    println!("✅ Engine initialized successfully!");
    println!();

    // Create a conversation with real-time processing
    println!("🎯 Starting STT conversation...");
    println!("💡 Say 'syrup' to activate, then speak your message");
    println!("🛑 Press Ctrl+C to stop");
    println!();

    let mut conversation_stream = stt_engine
        .conversation()
        .listen(|result| async move {
            match result {
                Ok(conversation) => {
                    println!("🟢 STT Conversation started successfully");
                    Ok(conversation.into_stream())
                }
                Err(e) => {
                    eprintln!("❌ Failed to start conversation: {}", e);
                    Err(e)
                }
            }
        })
        .await?;

    // Process the transcript stream
    while let Some(result) = conversation_stream.next().await {
        match result {
            Ok(segment) => {
                let timestamp = if segment.end_time() > segment.start_time() {
                    format!("{}ms", segment.end_time() - segment.start_time())
                } else {
                    "0ms".to_string()
                };

                if segment.text().contains("Wake word") {
                    println!("🔍 {}", segment.text());
                } else {
                    println!("📝 [{}] {}", timestamp, segment.text());
                }
            }
            Err(e) => {
                eprintln!("⚠️  Processing error: {}", e);
                // Continue processing despite errors
            }
        }
    }

    println!("👋 STT Demo completed");
    Ok(())
}
