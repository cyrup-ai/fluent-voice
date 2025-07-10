//! Showcase of the new ElevenLabs API builder chains.
//!
//! This example demonstrates all the new fluent builder chains
//! that expose ElevenLabs functionality behind the fluent-voice API.

use fluent_voice::prelude::*;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    println!("ElevenLabs API Showcase - All New Builder Chains");

    // 1. Voice Discovery
    println!("\n1. Voice Discovery:");
    let voices = FluentVoice::voices()
        .search("female")
        .category(VoiceCategory::Premade)
        .language(Language::ENGLISH_US)
        .labels(VoiceLabels::new().gender("female").age("young"))
        .page_size(10)
        .discover(|result| match result {
            Ok(voices) => {
                println!("   Found {} voices", voices.len());
                Ok(voices)
            }
            Err(e) => {
                println!("   Error: {}", e);
                Err(e)
            }
        })
        .await?;

    // 2. Voice Cloning
    println!("\n2. Voice Cloning:");
    let cloned_voice = FluentVoice::clone_voice()
        .from_samples(vec!["sample1.wav", "sample2.wav"])
        .name("MyCustomVoice")
        .description("A custom voice for narration")
        .labels(VoiceLabels::new().use_case("narration").gender("neutral"))
        .fine_tuning_model(ModelId::MultilingualV2)
        .create(|result| match result {
            Ok(voice) => {
                println!("   Created voice: {} ({})", voice.name(), voice.id().id());
                Ok(voice)
            }
            Err(e) => {
                println!("   Error: {}", e);
                Err(e)
            }
        })
        .await?;

    // 3. Speech-to-Speech Conversion
    println!("\n3. Speech-to-Speech Conversion:");
    let converted_audio = FluentVoice::speech_to_speech()
        .from_audio("input.wav")
        .target_voice(cloned_voice.voice_id.clone())
        .preserve_emotion(true)
        .preserve_style(true)
        .model(ModelId::MultilingualV2)
        .convert(|session| match session {
            Ok(session) => {
                println!("   Speech conversion initiated");
                Ok(session.into_stream())
            }
            Err(e) => {
                println!("   Error: {}", e);
                Err(e)
            }
        })
        .await?;

    // 4. Enhanced TTS with new features
    println!("\n4. Enhanced TTS:");
    let enhanced_audio = FluentVoice::tts()
        .with_speaker(
            Speaker::speaker("Alice")
                .voice_id(VoiceId::new("voice_123"))
                .speak("Hello! This demonstrates the enhanced TTS features.")
                .build(),
        )
        .output_format(AudioFormat::Mp3Khz44_192)
        .pronunciation_dictionary(PronunciationDictId::new("dict_123"))
        .seed(42)
        .previous_text("This was said before.")
        .synthesize(|conversation| match conversation {
            Ok(conversation) => {
                println!("   Enhanced TTS synthesis initiated");
                Ok(conversation.into_stream())
            }
            Err(e) => {
                println!("   Error: {}", e);
                Err(e)
            }
        })
        .await?;

    // 5. Audio Isolation
    println!("\n5. Audio Isolation:");
    let isolated_audio = FluentVoice::audio_isolation()
        .from_file("mixed_audio.wav")
        .isolate_voices(true)
        .remove_background(true)
        .isolation_strength(0.8)
        .process(|session| match session {
            Ok(session) => {
                println!("   Audio isolation processing initiated");
                Ok(session.into_stream())
            }
            Err(e) => {
                println!("   Error: {}", e);
                Err(e)
            }
        })
        .await?;

    // 6. Sound Effects Generation
    println!("\n6. Sound Effects Generation:");
    let sound_effects = FluentVoice::sound_effects()
        .describe("thunderstorm with heavy rain and distant thunder")
        .duration_seconds(30.0)
        .intensity(0.8)
        .mood("dramatic")
        .environment("outdoor")
        .seed(42)
        .generate(|session| match session {
            Ok(session) => {
                println!("   Sound effects generation initiated");
                Ok(session.into_stream())
            }
            Err(e) => {
                println!("   Error: {}", e);
                Err(e)
            }
        })
        .await?;

    println!("\n✅ All ElevenLabs API builder chains working successfully!");
    println!("   - Voice discovery with search and filtering");
    println!("   - Voice cloning from audio samples");
    println!("   - Speech-to-speech voice conversion");
    println!("   - Enhanced TTS with context, dictionaries, and formats");
    println!("   - Audio isolation and processing");
    println!("   - AI-powered sound effects generation");

    Ok(())
}
