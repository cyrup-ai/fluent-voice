//! ElevenLabs STT Example with Real File Transcription
//!
//! This example performs REAL speech-to-text transcription using ElevenLabs API.
//! It transcribes an audio file and shows real transcription results.

use fluent_voice_elevenlabs::*;
use std::path::Path;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎤 ElevenLabs STT Example with Real File Transcription");
    println!("=====================================================");
    println!("🎤 This performs REAL speech-to-text using ElevenLabs API");
    println!();

    // Check if audio file exists (you can create one or use any audio file)
    let audio_file = "test_audio.wav";
    if !Path::new(audio_file).exists() {
        println!("⚠️  Audio file '{}' not found.", audio_file);
        println!("⚠️  Please provide an audio file to transcribe.");
        println!("⚠️  You can record audio or use any .wav, .mp3, .m4a file.");
        println!();
        println!("📝 Example: Record audio with:");
        println!("   ffmpeg -f avfoundation -i ':0' -t 10 test_audio.wav");
        println!("   (or use any audio file you have)");
        return Ok(());
    }

    println!("📁 Transcribing file: {}", audio_file);
    println!();

    // Real file transcription with ElevenLabs
    let _transcript_output = FluentVoice::stt()
        .api_key_from_env()?
        .http3_enabled(true)
        .transcribe(audio_file)?
        .model("eleven_multilingual_v2")
        .language("en")
        .with_word_timestamps()
        .diarization(true)
        .tag_audio_events(true)
        .emit(|result| {
            match result {
                Ok(transcript_output) => {
                    println!("🔥 Real transcription results:");
                    println!("================================");
                    println!();

                    if transcript_output.text.is_empty() {
                        println!("⚠️  No text found in transcription.");
                        println!("⚠️  The audio file might be silent or too quiet.");
                    } else {
                        println!("🗣️  Transcribed text: {}", transcript_output.text);
                        println!("🎯 Overall confidence: {:.1}%", transcript_output.confidence * 100.0);

                        if !transcript_output.language.is_empty() {
                            println!("🌍 Detected language: {}", transcript_output.language);
                        }

                        // Note: ElevenLabs API currently returns text-only transcription
                        // Word-level timestamps are not available through this interface
                        if transcript_output.words.is_empty() {
                            println!("📝 Note: Word-level breakdown not available with current ElevenLabs API");
                        } else {
                            println!("🔍 Word-by-word breakdown:");
                            for (i, word) in transcript_output.words.iter().enumerate() {
                                println!("{:2}. \"{}\"", i + 1, word.text);
                            }
                        }
                    }

                    println!();
                    println!("✅ Real ElevenLabs STT transcription completed!");
                    Ok(transcript_output)
                }
                Err(e) => {
                    println!("❌ Transcription failed: {}", e);
                    Err(e)
                }
            }
        }).await?;

    println!("📊 Successfully transcribed audio file: '{}'", audio_file);

    Ok(())
}
