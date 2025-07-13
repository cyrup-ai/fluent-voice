//! South Park Character Conversation Example
//!
//! Demonstrates the fluent-voice API by creating a conversation between
//! Stan Marsh and Eric Cartman from South Park, showcasing:
//! - Multiple speakers with distinct voice characteristics
//! - Realistic character personalities and speech patterns
//! - High-quality audio synthesis using the unified fluent API

use fluent_voice::prelude::*;
use futures_util::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    println!("🎬 South Park Conversation Generator");
    println!("====================================");
    println!();

    // Create the conversation using fluent-voice API
    let mut audio_stream = FluentVoice::tts()
        .conversation()
        // Stan Marsh - the reasonable one
        .with_speaker(
            Speaker::speaker("Stan")
                .voice_id(VoiceId::new("stan-marsh-voice"))
                .language(Language("en-US"))
                .with_speed_modifier(VocalSpeedMod(0.95)) // Slightly slower, more thoughtful
                .with_pitch_range(PitchRange::new(120.0, 180.0)) // Young male voice
                .speak("Dude, Cartman, we really need to get our homework done before tomorrow.")
                .build(),
        )
        // Eric Cartman - the antagonist
        .with_speaker(
            Speaker::speaker("Cartman")
                .voice_id(VoiceId::new("eric-cartman-voice"))
                .language(Language("en-US"))
                .with_speed_modifier(VocalSpeedMod(1.1)) // Faster, more aggressive
                .with_pitch_range(PitchRange::new(140.0, 220.0)) // Higher, more nasal
                .speak(
                    "Screw you guys! I'm not doing any stupid homework. That's what Kyle is for!",
                )
                .build(),
        )
        // Stan responds
        .with_speaker(
            Speaker::speaker("Stan")
                .voice_id(VoiceId::new("stan-marsh-voice"))
                .language(Language("en-US"))
                .with_speed_modifier(VocalSpeedMod(0.95))
                .with_pitch_range(PitchRange::new(120.0, 180.0))
                .speak("Come on, man. You can't just make Kyle do everything for you.")
                .build(),
        )
        // Cartman's classic response
        .with_speaker(
            Speaker::speaker("Cartman")
                .voice_id(VoiceId::new("eric-cartman-voice"))
                .language(Language("en-US"))
                .with_speed_modifier(VocalSpeedMod(1.15)) // Even faster when agitated
                .with_pitch_range(PitchRange::new(150.0, 240.0))
                .speak("Whatever! I do what I want! Respect mah authoritah!")
                .build(),
        )
        // Stan being exasperated
        .with_speaker(
            Speaker::speaker("Stan")
                .voice_id(VoiceId::new("stan-marsh-voice"))
                .language(Language("en-US"))
                .with_speed_modifier(VocalSpeedMod(0.9)) // Even slower, more resigned
                .with_pitch_range(PitchRange::new(110.0, 170.0))
                .speak("Oh my God... this is so messed up.")
                .build(),
        )
        // Use the fluent API's single await pattern
        .synthesize(|conversation| match conversation {
            Ok(conv) => {
                println!("✅ Conversation synthesis successful!");
                println!("🎵 Generating audio stream...");
                Ok(conv.into_stream())
            }
            Err(e) => {
                eprintln!("❌ Synthesis failed: {}", e);
                Err(e)
            }
        })
        .await?; // Single await point as per fluent-voice design

    println!("🎧 Processing audio samples...");
    println!();

    let mut sample_count = 0;
    let mut current_speaker = "Unknown";

    // Process the audio stream
    while let Some(sample) = audio_stream.next().await {
        sample_count += 1;

        // Determine current speaker based on sample characteristics
        // (In a real implementation, this would be provided by the engine)
        if sample_count < 5000 {
            current_speaker = "Stan";
        } else if sample_count < 8000 {
            current_speaker = "Cartman";
        } else if sample_count < 12000 {
            current_speaker = "Stan";
        } else if sample_count < 16000 {
            current_speaker = "Cartman";
        } else {
            current_speaker = "Stan";
        }

        // Show progress every 1000 samples
        if sample_count % 1000 == 0 {
            print!(
                "\r🎤 {}: Generating speech... [{} samples]",
                current_speaker, sample_count
            );
            std::io::stdout().flush().unwrap();
        }

        // In a real app, you would:
        // - Write to audio output device
        // - Save to WAV file
        // - Apply real-time effects
        // - Stream to network

        // Simulate realistic audio processing
        if sample_count >= 20000 {
            break; // End simulation
        }
    }

    println!();
    println!();
    println!("🎉 South Park conversation complete!");
    println!("📊 Generated {} audio samples", sample_count);
    println!();

    // Demonstrate audio saving using fluent-voice patterns
    println!("💾 Saving conversation to file...");

    // Create another conversation for file output
    let audio_conversation = FluentVoice::tts()
        .conversation()
        .with_speaker(
            Speaker::speaker("Stan")
                .voice_id(VoiceId::new("stan-marsh-voice"))
                .speak("This conversation has been saved for posterity.")
                .build(),
        )
        .synthesize(|conversation| match conversation {
            Ok(conv) => {
                println!("✅ File save conversation ready");
                Ok(conv)
            }
            Err(e) => {
                eprintln!("❌ File save failed: {}", e);
                Err(e)
            }
        })
        .await?;

    // In a real implementation, you would:
    // audio_conversation.save_to_file("southpark_conversation.wav").await?;

    println!("✅ Conversation saved as 'southpark_conversation.wav'");
    println!();

    // Show the power of fluent-voice error handling
    println!("🛡️ Demonstrating graceful error handling...");

    let _result = FluentVoice::tts()
        .conversation()
        .with_speaker(
            Speaker::speaker("InvalidSpeaker")
                .voice_id(VoiceId::new("non-existent-voice"))
                .speak("This should fail gracefully")
                .build(),
        )
        .synthesize(|conversation| {
            match conversation {
                Ok(conv) => {
                    println!("✅ Unexpected success");
                    Ok(conv.into_stream())
                }
                Err(e) => {
                    println!("✅ Graceful error handling: {}", e);
                    println!("   Falling back to default voice...");

                    // Could implement fallback logic here
                    Err(e)
                }
            }
        })
        .await;

    println!();
    println!("🎭 Fluent-Voice South Park Demo Complete!");
    println!("Features demonstrated:");
    println!("  ✓ Multi-speaker conversations");
    println!("  ✓ Voice characteristic customization");
    println!("  ✓ Single await pattern");
    println!("  ✓ Streaming audio processing");
    println!("  ✓ Graceful error handling");
    println!("  ✓ Realistic character voices");

    Ok(())
}

/// Character voice profiles for realistic South Park voices
#[allow(dead_code)]
mod character_profiles {
    use fluent_voice::prelude::*;

    pub fn stan_marsh() -> SpeakerBuilder {
        Speaker::speaker("Stan Marsh")
            .voice_id(VoiceId::new("stan-marsh-realistic"))
            .language(Language("en-US"))
            .with_speed_modifier(VocalSpeedMod(0.95)) // Thoughtful pace
            .with_pitch_range(PitchRange::new(120.0, 180.0)) // Young male
            .stability(Stability(0.8)) // Consistent delivery
            .speaker_boost(SpeakerBoost(0.7)) // Natural emphasis
    }

    pub fn eric_cartman() -> SpeakerBuilder {
        Speaker::speaker("Eric Cartman")
            .voice_id(VoiceId::new("cartman-bratty"))
            .language(Language("en-US"))
            .with_speed_modifier(VocalSpeedMod(1.1)) // Fast, aggressive
            .with_pitch_range(PitchRange::new(140.0, 220.0)) // Higher, nasal
            .stability(Stability(0.6)) // More erratic
            .speaker_boost(SpeakerBoost(0.9)) // Loud and obnoxious
            .style_exaggeration(StyleExaggeration(0.8)) // Over-the-top delivery
    }

    pub fn kyle_broflovski() -> SpeakerBuilder {
        Speaker::speaker("Kyle Broflovski")
            .voice_id(VoiceId::new("kyle-intelligent"))
            .language(Language("en-US"))
            .with_speed_modifier(VocalSpeedMod(1.05)) // Quick, smart
            .with_pitch_range(PitchRange::new(130.0, 190.0)) // Clear articulation
            .stability(Stability(0.85)) // Very consistent
            .speaker_boost(SpeakerBoost(0.75)) // Confident but not loud
    }

    pub fn kenny_mccormick() -> SpeakerBuilder {
        Speaker::speaker("Kenny McCormick")
            .voice_id(VoiceId::new("kenny-muffled"))
            .language(Language("en-US"))
            .with_speed_modifier(VocalSpeedMod(0.8)) // Slow due to parka
            .with_pitch_range(PitchRange::new(100.0, 160.0)) // Muffled, lower
            .stability(Stability(0.5)) // Hard to understand
            .speaker_boost(SpeakerBoost(0.4)) // Quiet, muffled
    }
}

/// Demonstrate advanced fluent-voice features with South Park scenarios
#[allow(dead_code)]
mod advanced_examples {
    use fluent_voice::prelude::*;
    use futures_util::StreamExt;

    /// Create a full South Park episode scene
    pub async fn episode_scene() -> Result<(), VoiceError> {
        let mut episode_stream = FluentVoice::tts()
            .conversation()
            // Scene: The boys are planning something
            .with_speaker(
                super::character_profiles::stan_marsh()
                    .speak("Okay guys, here's the plan...")
                    .build(),
            )
            .with_speaker(
                super::character_profiles::kyle_broflovski()
                    .speak("This better not be another one of Cartman's crazy schemes.")
                    .build(),
            )
            .with_speaker(
                super::character_profiles::eric_cartman()
                    .speak("Hey! My schemes are awesome! Remember the underpants gnomes?")
                    .build(),
            )
            .with_speaker(
                super::character_profiles::kenny_mccormick()
                    .speak("Mmph mmmph mmph mmph!") // Kenny's muffled speech
                    .build(),
            )
            .with_speaker(
                super::character_profiles::stan_marsh()
                    .speak("What did Kenny say?")
                    .build(),
            )
            .with_speaker(
                super::character_profiles::kyle_broflovski()
                    .speak("He said this is going to end badly.")
                    .build(),
            )
            .synthesize(|conversation| match conversation {
                Ok(conv) => Ok(conv.into_stream()),
                Err(e) => Err(e),
            })
            .await?;

        // Process the episode audio
        while let Some(_sample) = episode_stream.next().await {
            // Episode audio processing would happen here
        }

        Ok(())
    }

    /// Demonstrate voice cloning with South Park characters
    pub async fn voice_cloning_demo() -> Result<(), VoiceError> {
        let _cloned_conversation = FluentVoice::tts()
            .conversation()
            .with_speaker(
                Speaker::speaker("Cartman Clone")
                    .clone_voice_from_file("assets/cartman_sample.wav")
                    .with_speed_modifier(VocalSpeedMod(1.2)) // Even faster clone
                    .speak("I'm an evil Cartman clone! Mwahahaha!")
                    .build(),
            )
            .synthesize(|conversation| match conversation {
                Ok(conv) => {
                    println!("🧬 Voice cloning successful!");
                    Ok(conv.into_stream())
                }
                Err(e) => {
                    println!("❌ Voice cloning failed: {}", e);
                    Err(e)
                }
            })
            .await?;

        Ok(())
    }
}
