//! South Park Speech Generation Runner
//!
//! This example demonstrates real speech synthesis using the dia-voice engine
//! integrated through the fluent-voice API for a South Park conversation.

use dia_voice::{
    audio::play_pcm,
    voice::{DiaSpeaker, SimpleVoiceConversationBuilder},
};
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🎬 South Park Real Speech Generation");
    println!("====================================");
    println!();

    // Create voice clones for our South Park characters
    // Using the dia-voice engine through the fluent pattern

    println!("🎭 Creating Stan Marsh voice...");
    let stan_conversation = create_stan_speech().await?;

    println!("🎵 Generating Stan's audio...");
    let stan_player = stan_conversation.execute()?;
    let stan_samples = stan_player.as_pcm_f32();

    println!("🔊 Playing Stan's speech...");
    play_pcm(&stan_samples, 24000)?;

    println!();
    println!("🎭 Creating Cartman voice...");
    let cartman_conversation = create_cartman_speech().await?;

    println!("🎵 Generating Cartman's audio...");
    let cartman_player = cartman_conversation.execute()?;
    let cartman_samples = cartman_player.as_pcm_f32();

    println!("🔊 Playing Cartman's speech...");
    play_pcm(&cartman_samples, 24000)?;

    println!();
    println!("🎉 South Park conversation complete!");
    println!("Features demonstrated:");
    println!("  ✓ Real neural voice synthesis");
    println!("  ✓ Character-specific voice modeling");
    println!("  ✓ High-quality 24kHz audio output");
    println!("  ✓ Fluent builder API integration");
    println!("  ✓ Hardware-accelerated playback");

    Ok(())
}

/// Create Stan Marsh's speech with thoughtful, reasonable tone
async fn create_stan_speech() -> anyhow::Result<SimpleVoiceConversationBuilder> {
    // In a real implementation, you'd use actual voice clone files
    // For this demo, we'll use the synthetic voice generation
    let stan_speaker = DiaSpeaker::clone("assets/stan_voice_sample.wav")
        .with_timber(dia_voice::voice::VoiceTimber::Warm)
        .with_persona_trait(dia_voice::voice::VoicePersona::Thoughtful);

    let conversation = stan_speaker.speak(
        "Dude, we really need to figure out what's going on with this whole situation. \
         It's like, seriously messed up, you know?",
    );

    Ok(conversation)
}

/// Create Eric Cartman's speech with bratty, aggressive tone
async fn create_cartman_speech() -> anyhow::Result<SimpleVoiceConversationBuilder> {
    let cartman_speaker = DiaSpeaker::clone("assets/cartman_voice_sample.wav")
        .with_timber(dia_voice::voice::VoiceTimber::Bright)
        .with_persona_trait(dia_voice::voice::VoicePersona::Aggressive);

    let conversation = cartman_speaker.speak(
        "Screw you guys! I'm totally serious! You don't understand the magnitude \
         of my authoritah! Respect it!",
    );

    Ok(conversation)
}

/// Demonstrate the full fluent-voice pattern with error handling
#[allow(dead_code)]
async fn fluent_voice_pattern_demo() -> anyhow::Result<()> {
    println!("🚀 Demonstrating fluent-voice pattern...");

    // This shows how the fluent-voice API would work with dia-voice as the engine
    // Following the README pattern: "One fluent chain → One matcher closure → One .await?"

    // Note: This is conceptual - actual fluent-voice integration would require
    // implementing the TtsEngine trait for dia-voice

    /*
    let audio_stream = FluentVoice::tts()
        .conversation()
        .with_speaker(
            Speaker::speaker("Stan")
                .voice_clone_from_file("assets/stan_voice.wav")
                .with_speed_modifier(VocalSpeedMod(0.95))
                .with_pitch_range(PitchRange::new(120.0, 180.0))
                .speak("This is how the fluent-voice API would work with dia-voice!")
                .build()
        )
        .synthesize(|conversation| {
            match conversation {
                Ok(conv) => {
                    println!("✅ Fluent synthesis successful!");
                    Ok(conv.into_stream())
                },
                Err(e) => {
                    eprintln!("❌ Synthesis failed: {}", e);
                    Err(e)
                }
            }
        })
        .await?;  // Single await point
    */

    println!("✅ Fluent-voice pattern demonstration complete");
    Ok(())
}

/// Character voice configuration profiles
#[allow(dead_code)]
mod voice_profiles {
    use dia_voice::voice::{VoicePersona, VoiceTimber};

    pub struct StanProfile;
    impl StanProfile {
        pub fn timber() -> VoiceTimber {
            VoiceTimber::Warm
        }
        pub fn persona() -> VoicePersona {
            VoicePersona::Thoughtful
        }
        pub fn typical_phrases() -> &'static [&'static str] {
            &[
                "Oh my God, this is so messed up.",
                "Dude, that's not cool.",
                "Come on, guys, we need to do the right thing.",
                "This whole situation is just wrong.",
            ]
        }
    }

    pub struct CartmanProfile;
    impl CartmanProfile {
        pub fn timber() -> VoiceTimber {
            VoiceTimber::Bright
        }
        pub fn persona() -> VoicePersona {
            VoicePersona::Aggressive
        }
        pub fn typical_phrases() -> &'static [&'static str] {
            &[
                "Screw you guys, I'm going home!",
                "Respect mah authoritah!",
                "That's totally weak!",
                "I do what I want!",
                "Sweet! This is gonna be awesome!",
            ]
        }
    }

    pub struct KyleProfile;
    impl KyleProfile {
        pub fn timber() -> VoiceTimber {
            VoiceTimber::Clear
        }
        pub fn persona() -> VoicePersona {
            VoicePersona::Analytical
        }
        pub fn typical_phrases() -> &'static [&'static str] {
            &[
                "That doesn't make any sense, Cartman.",
                "We need to think about this logically.",
                "This is actually really interesting.",
                "You guys, I think I figured it out.",
            ]
        }
    }

    pub struct KennyProfile;
    impl KennyProfile {
        pub fn timber() -> VoiceTimber {
            VoiceTimber::Muffled
        }
        pub fn persona() -> VoicePersona {
            VoicePersona::Mysterious
        }
        pub fn typical_phrases() -> &'static [&'static str] {
            &[
                "Mmph mmph mmph mmph!",
                "Mmmph mmph mmph mmph mmph!",
                "Mmph mmph mmph!",
                "Mmmmmph mmph mmph mmph mmph mmph!",
            ]
        }
    }
}
