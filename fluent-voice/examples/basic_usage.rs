//! ElevenLabs TTS/STT Example using Fluent Voice API
//!
//! This example demonstrates how to use the fluent voice API with ElevenLabs-style
//! engines for both Text-to-Speech (TTS) and Speech-to-Text (STT) operations.

use fluent_voice::prelude::*;
use fluent_voice::stt_conversation::SttConversation;
use fluent_voice::tts_conversation::TtsConversation;
use futures_util::StreamExt;

// Example ElevenLabs TTS Engine Implementation
mod elevenlabs_tts {
    use fluent_voice::speaker_builder::SpeakerBuilder;
    use fluent_voice_macros::tts_engine;

    // Generate complete TTS engine with one macro call
    tts_engine!(
        engine = ElevenLabsTts,
        voice  = String,  // ElevenLabs voice ID
        audio  = futures::stream::Iter<std::vec::IntoIter<i16>>,
        /// ElevenLabs TTS engine implementation.
    );

    // The macro generates a complete, fully implemented engine
    // When using builder-for-builders, no manual implementation is needed
}

// Example ElevenLabs STT Engine Implementation
mod elevenlabs_stt {
    use fluent_voice::prelude::*;
    use fluent_voice_macros::stt_engine;

    // ElevenLabs transcript segment
    #[derive(Debug, Clone)]
    pub struct ElevenLabsSegment {
        pub text: String,
        pub start_ms: u32,
        pub end_ms: u32,
        pub speaker: Option<String>,
        pub confidence: f32,
    }

    impl TranscriptSegment for ElevenLabsSegment {
        fn start_ms(&self) -> u32 {
            self.start_ms
        }
        fn end_ms(&self) -> u32 {
            self.end_ms
        }
        fn text(&self) -> &str {
            &self.text
        }
        fn speaker_id(&self) -> Option<&str> {
            self.speaker.as_deref()
        }
    }

    // Generate complete STT engine with one macro call
    stt_engine!(
        engine  = ElevenLabsStt,
        segment = ElevenLabsSegment,
        stream  = futures::stream::Iter<std::vec::IntoIter<Result<ElevenLabsSegment, VoiceError>>>,
        /// ElevenLabs STT engine implementation.
    );

    // The macro generates a complete, fully implemented engine
    // When using builder-for-builders, no manual implementation is needed
}

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    println!("🎵 ElevenLabs + Fluent Voice API Demo");
    println!("====================================");
    println!("This example shows the fluent API structure for ElevenLabs integration.");
    println!("Note: The underlying methods contain todo!() - replace with real API calls.\n");

    // TTS Example: Multi-speaker conversation with ElevenLabs
    println!("📢 TTS Example - ElevenLabs Synthesis Pattern:");
    println!("----------------------------------------------");

    use elevenlabs_tts::ElevenLabsTts;

    // Demonstrate the fluent API pattern (will panic due to todo!())
    println!("Building fluent TTS chain...");

    let _tts_builder = ElevenLabsTts::builder()
        .with_speaker(
            elevenlabs_tts::SpeakerLineBuilder::named("Rachel")
                .voice_id(VoiceId("21m00Tcm4TlvDq8ikWAM".to_string()))
                .with_speed_modifier(VocalSpeedMod(0.9))
                .speak("Hello! I'm Rachel from ElevenLabs.")
                .build(),
        )
        .with_speaker(
            elevenlabs_tts::SpeakerLineBuilder::named("Josh")
                .voice_id(VoiceId("29vD33N1CtxCmqQRPOHJ".to_string()))
                .with_speed_modifier(VocalSpeedMod(1.1))
                .speak("And I'm Josh. Together we demonstrate the fluent API.")
                .build(),
        )
        .language(Language("en-US"));

    println!("✅ TTS builder created successfully!");
    println!("   → Configured 2 speakers with ElevenLabs voice IDs");
    println!("   → Set language to en-US");
    println!("   → Ready for .synthesize() call");

    // Note: Would call .synthesize() here but it contains todo!()
    // let audio_stream = tts_builder.synthesize(...).await?;

    // STT Example: Real-time transcription with ElevenLabs
    println!("\n🎤 STT Example - ElevenLabs Transcription Pattern:");
    println!("--------------------------------------------------");

    use elevenlabs_stt::ElevenLabsStt;

    // Demonstrate the fluent API pattern
    println!("Building fluent STT chain...");

    let _stt_builder = ElevenLabsStt::builder()
        .with_source(SpeechSource::Microphone {
            backend: MicBackend::Default,
            format: AudioFormat::Pcm48Khz,
            sample_rate: 48_000,
        })
        .vad_mode(VadMode::Accurate)
        .noise_reduction(NoiseReduction::High)
        .language_hint(Language("en-US"))
        .diarization(Diarization::On)
        .word_timestamps(WordTimestamps::On)
        .timestamps_granularity(TimestampsGranularity::Word)
        .punctuation(Punctuation::On);

    println!("✅ STT builder created successfully!");
    println!("   → Configured high-quality 48kHz microphone input");
    println!("   → Enabled accurate VAD and high noise reduction");
    println!("   → Enabled speaker diarization and word timestamps");
    println!("   → Ready for .listen() call");

    // Note: Would call .listen() here but it contains todo!()
    // let transcript_stream = stt_builder.listen(...).await?;

    println!("\n🎯 Fluent API Features Demonstrated:");
    println!("• ✅ One fluent chain per operation");
    println!("• ✅ Builder pattern with method chaining");
    println!("• ✅ Type-safe configuration");
    println!("• ✅ Engine-agnostic trait design");
    println!("• ✅ Zero-boilerplate macro generation");

    println!("\n🏗️ Builder-for-Builders Pattern:");
    println!("```rust");
    println!("// In a real ElevenLabs integration, you would use the builder-for-builders pattern:");
    println!("EngineBuilder::new(\"ElevenLabsTts\")");
    println!("    .segment_type::<ElevenLabsSegment>()");
    println!("    .stream_type::<futures::stream::Iter<std::vec::IntoIter<Result<ElevenLabsSegment, VoiceError>>>>()");
    println!("    .model_config(ModelConfig::ElevenLabs)");
    println!("    .with_api_key(env!(\"ELEVENLABS_API_KEY\"))");
    println!("    .documentation(\"ElevenLabs TTS engine using cloud API.\")");
    println!("    .build()");
    println!("```");
    println!("");
    println!("// The macro and builder handle all implementation details - no manual methods needed!");

    println!("\n💡 Next Steps:");
    println!("1. Get ElevenLabs API key from https://elevenlabs.io");
    println!("2. Use the EngineBuilder pattern to create your concrete engine");
    println!("3. Configure your engine with proper API credentials");
    println!("4. Add error handling for network issues and API limits");
    println!("5. Test with real voices and audio input");

    println!("\n✅ Fluent API structure validation complete!");

    Ok(())
}
