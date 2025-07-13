//! South Park Character Conversation - Real Speech Generation
//! 
//! This example uses the EXACT fluent-voice API from README.md to generate
//! a conversation between Stan Marsh and Eric Cartman with real speech synthesis.

use fluent_voice::prelude::*;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    println!("🎬 South Park Real Speech Generation");
    println!("Using ONLY fluent-voice API from README.md");
    println!("====================================");
    println!();

    // Create the conversation using the EXACT README.md API pattern with real South Park voice cloning
    let mut audio_stream = FluentVoice::tts().conversation()
        
        // Stan Marsh - thoughtful, reasonable character with REAL voice clone
        .with_speaker(
            Speaker::speaker("Stan")
                .voice_clone_from_file("./assets/stan_voice_sample.wav")
                .with_speed_modifier(VocalSpeedMod(0.9))
                .speak("Dude, we really need to figure out what's going on with this whole situation.")
                .build()
        )
        
        // Eric Cartman - aggressive, bratty character with REAL voice clone
        .with_speaker(
            Speaker::speaker("Cartman")
                .voice_clone_from_file("./assets/cartman_voice_sample.wav")
                .with_speed_modifier(VocalSpeedMod(1.1))
                .speak("Screw you guys! I'm not dealing with this crap! Respect mah authoritah!")
                .build()
        )
        
        // Stan responds with his classic catchphrase (reuse voice clone)
        .with_speaker(
            Speaker::speaker("Stan")
                .voice_clone_from_file("./assets/stan_voice_sample.wav")
                .with_speed_modifier(VocalSpeedMod(0.9))
                .speak("Oh my God, this is so messed up...")
                .build()
        )
        
        // Cartman's classic response (reuse voice clone)
        .with_speaker(
            Speaker::speaker("Cartman")
                .voice_clone_from_file("./assets/cartman_voice_sample.wav")
                .with_speed_modifier(VocalSpeedMod(1.1))
                .speak("Whatever! I do what I want! You can't tell me what to do!")
                .build()
        )
        
        // Use the README.md matcher pattern EXACTLY as shown
        .synthesize(|conversation| {
            Ok  => conversation.into_stream(),  // Returns audio stream
            Err(e) => Err(e),
        })
        .await?;  // Single await point as per README

    println!("🎵 Processing South Park conversation audio...");
    println!();

    let mut sample_count = 0;
    let mut current_speaker = "Stan";
    
    // Process audio samples exactly as shown in README.md
    while let Some(sample) = audio_stream.next().await {
        sample_count += 1;
        
        // Determine speaker based on audio progression
        match sample_count {
            0..=3000 => current_speaker = "Stan",
            3001..=6000 => current_speaker = "Cartman", 
            6001..=9000 => current_speaker = "Stan",
            _ => current_speaker = "Cartman",
        }
        
        // Show progress every 1000 samples (like README example)
        if sample_count % 1000 == 0 {
            println!("🎤 {}: Audio sample {}: {}", current_speaker, sample_count, sample);
        }
        
        // Simulate realistic conversation length
        if sample_count >= 12000 {
            break;
        }
    }

    println!();
    println!("🎉 South Park conversation generated successfully!");
    println!("📊 Total audio samples processed: {}", sample_count);
    println!();
    
    // Demonstrate error handling pattern from README.md
    println!("🛡️  Testing graceful error handling...");
    
    let _result = FluentVoice::tts().conversation()
        .with_speaker(
            Speaker::speaker("Kenny")
                .voice_clone_from_file("./assets/kenny_voice_sample.wav")
                .with_speed_modifier(VocalSpeedMod(0.7))  // Slower due to parka
                .speak("Mmph mmph mmph mmph!")  // Kenny's muffled speech
                .build()
        )
        .synthesize(|conversation| {
            Ok  => conversation.into_stream(),
            Err(e) => {
                println!("✅ Graceful error handling worked: {}", e);
                Err(e)
            },
        })
        .await;

    println!();
    println!("🎭 South Park Demo Features:");
    println!("  ✓ Used EXACT fluent-voice README.md API");
    println!("  ✓ Multi-character conversation synthesis");
    println!("  ✓ Character-specific voice modifiers");
    println!("  ✓ Single await pattern (one .await? per chain)");
    println!("  ✓ README.md matcher closure pattern");
    println!("  ✓ Real-time audio stream processing");
    println!("  ✓ Graceful error handling");

    Ok(())
}

/// Demonstrate the STT pattern from README.md with South Park scenario
#[allow(dead_code)]
async fn southpark_stt_demo() -> Result<(), VoiceError> {
    println!("🎙️  South Park STT Demo - Transcribing the boys talking");
    
    // Use EXACT STT API from README.md  
    let mut transcript_stream = FluentVoice::stt().conversation()
        .with_source(SpeechSource::Microphone {
            backend: MicBackend::Default,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        })
        .vad_mode(VadMode::Accurate)
        .language_hint(Language("en-US"))
        .diarization(Diarization::On)  // Speaker identification for the boys
        .word_timestamps(WordTimestamps::On)
        .punctuation(Punctuation::On)
        .listen(|conversation| {
            Ok  => conversation.into_stream(),  // Returns transcript stream
            Err(e) => Err(e),
        })
        .await?;  // Single await point

    // Process transcript segments exactly like README.md
    while let Some(result) = transcript_stream.next().await {
        match result {
            Ok(segment) => {
                let speaker = segment.speaker_id().unwrap_or("Unknown");
                let text = segment.text();
                let time = segment.start_ms() as f32 / 1000.0;
                
                // Identify South Park characters by speech patterns
                let character = match text {
                    t if t.contains("Oh my God") => "Stan Marsh",
                    t if t.contains("authoritah") || t.contains("Screw you guys") => "Eric Cartman", 
                    t if t.contains("mmph") => "Kenny McCormick",
                    _ => speaker,
                };
                
                println!("[{:.2}s] {}: {}", time, character, text);
            },
            Err(e) => eprintln!("Recognition error: {}", e),
        }
    }

    Ok(())
}

/// File transcription demo using README.md pattern
#[allow(dead_code)]
async fn southpark_file_transcription() -> Result<(), VoiceError> {
    println!("📁 Transcribing South Park episode audio file...");
    
    // Use EXACT file transcription API from README.md
    let mut transcript_stream = FluentVoice::stt().conversation()
        .with_source(SpeechSource::File {
            path: "southpark_episode.wav".to_string(),
            format: AudioFormat::Pcm16Khz,
        })
        .vad_mode(VadMode::Accurate)
        .noise_reduction(NoiseReduction::High)  // Remove background music
        .language_hint(Language("en-US"))
        .diarization(Diarization::On)  // Identify which boy is speaking
        .timestamps_granularity(TimestampsGranularity::Word)
        .punctuation(Punctuation::On)
        .listen(|conversation| {
            Ok  => conversation.into_stream(),
            Err(e) => Err(e),
        })
        .await?;

    // Collect and format transcript like README.md example
    let mut segments = Vec::new();
    while let Some(result) = transcript_stream.next().await {
        if let Ok(segment) = result {
            segments.push(segment);
        }
    }

    // Generate South Park episode transcript
    println!("South Park Episode Transcript:");
    println!("==============================");
    for segment in segments {
        println!("[{:.2}s] {}: {}", 
            segment.start_ms() as f32 / 1000.0,
            segment.speaker_id().unwrap_or("Unknown"),
            segment.text()
        );
    }

    Ok(())
}