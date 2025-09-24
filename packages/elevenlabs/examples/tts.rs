//! Demonstrates the fluent builder API for Text-to-Speech (TTS) using ElevenLabs

use fluent_voice_elevenlabs::prelude::*;
use fluent_voice_elevenlabs::voice::Voice;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    println!("🔍 Starting ElevenLabs TTS test...");
    
    // Use ElevenLabsFluentVoice directly to get real ElevenLabs functionality
    let builder = ElevenLabsFluentVoice::tts()
        .await?
        .conversation()
        .with_speaker(
            Speaker::speaker("Narrator")
                .add_line("This is a demonstration of real-time text-to-speech synthesis.")
                .add_line("The audio is generated using ElevenLabs TTS engine.")
                .add_line("Thank you for listening!")
                .with_voice(Voice::Rachel) // Use proper voice enum
                .with_speed(1.0)
                .build(),
        );

    println!("🔧 Getting audio stream...");
    let mut stream = TtsConversationChunkBuilder::synthesize(builder);
    
    println!("🔍 Processing audio chunks...");
    let mut chunk_count = 0;
    let mut total_bytes = 0;
    
    while let Some(chunk) = stream.next().await {
        chunk_count += 1;
        let data_len = chunk.data.len();
        total_bytes += data_len;
        
        println!("📦 Chunk {}: {} bytes", chunk_count, data_len);
        
        if let Some(text) = &chunk.text {
            if text.starts_with("[ERROR]") {
                println!("   ❌ ERROR: {}", text);
            } else {
                println!("   Text: {}", text);
            }
        }
        
        if data_len > 0 {
            println!("   First 10 bytes: {:?}", &chunk.data[..std::cmp::min(10, data_len)]);
        } else {
            println!("   ⚠️  Empty chunk!");
        }
    }
    
    println!("📊 Summary: {} chunks, {} total bytes", chunk_count, total_bytes);
    
    if total_bytes > 0 {
        println!("✅ Audio data received - attempting playback...");
        // Now play the audio
        let builder = ElevenLabsFluentVoice::tts()
            .await?
            .conversation()
            .with_speaker(
                Speaker::speaker("Narrator")
                    .add_line("This is a demonstration of real-time text-to-speech synthesis.")
                    .with_voice(Voice::Rachel)
                    .build(),
            );
        
        TtsConversationChunkBuilder::synthesize(builder).play().await?;
    } else {
        println!("❌ No audio data received - check ElevenLabs API key and connectivity");
    }

    Ok(())
}
