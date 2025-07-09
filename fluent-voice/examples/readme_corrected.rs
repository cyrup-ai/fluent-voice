//! Corrected versions of README.md examples that work with the actual implementation

use fluent_voice::prelude::*;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), VoiceError> {
    println!("🎯 Corrected README Examples\n");

    // Corrected Example 1: STT with file transcription and collect()
    println!("Example 1: STT with transcribe() and collect()");
    {
        // The collect() method is available on TranscriptionBuilder, not after listen()
        let _transcript = FluentVoiceImpl::stt()
            .transcribe("audio.wav") // Use transcribe() for files
            .vad_mode(VadMode::Accurate)
            .language_hint(Language("en-US"))
            .diarization(Diarization::On)
            .word_timestamps(WordTimestamps::On)
            .punctuation(Punctuation::On)
            .collect() // Now collect() works!
            .await?;

        println!("✅ Transcription with collect() works correctly");
    }

    // Corrected Example 2: STT with microphone streaming
    println!("\nExample 2: STT with microphone and streaming");
    {
        // For live streaming, use listen() but handle the stream properly
        let stream = FluentVoiceImpl::stt()
            .with_source(SpeechSource::Microphone {
                backend: MicBackend::Default,
                format: AudioFormat::Pcm16Khz,
                sample_rate: 16_000,
            })
            .vad_mode(VadMode::Accurate)
            .language_hint(Language("en-US"))
            .diarization(Diarization::On)
            .word_timestamps(WordTimestamps::On)
            .timestamps_granularity(TimestampsGranularity::Word) // Include all methods
            .punctuation(Punctuation::On)
            .listen(|conversation| match conversation {
                // Correct matcher syntax
                Ok(conv) => Ok(conv.into_stream()),
                Err(e) => Err(e),
            })
            .await?;

        println!("✅ Microphone streaming setup works correctly");
    }

    // Corrected Example 3: TTS with proper syntax
    println!("\nExample 3: TTS with correct Speaker syntax");
    {
        // Using the actual types from our implementation
        let mut audio_stream = FluentVoiceImpl::tts()
            .with_speaker(
                SpeakerLine::speaker("Alice") // Correct type name
                    .voice_id(VoiceId("voice-uuid".to_string())) // Correct constructor
                    .with_speed_modifier(VocalSpeedMod(0.9))
                    .speak("Hello, world!")
                    .build(),
            )
            .with_speaker(
                SpeakerLine::speaker("Bob")
                    .voice_id(VoiceId("another-voice".to_string()))
                    .with_speed_modifier(VocalSpeedMod(1.1))
                    .speak("Hi Alice! How are you today?")
                    .build(),
            )
            .language(Language("en-US"))
            .model(ModelId("model-id".to_string())) // Show all available methods
            .stability(Stability(0.5))
            .similarity(Similarity(0.8))
            .speaker_boost(SpeakerBoost(true))
            .style_exaggeration(StyleExaggeration(0.3))
            .synthesize(|conversation| match conversation {
                // Correct matcher syntax
                Ok(conv) => Ok(conv.into_stream()),
                Err(e) => Err(e),
            })
            .await?;

        // Process audio samples
        let mut count = 0;
        while let Some(sample) = audio_stream.next().await {
            count += 1;
            if count > 3 {
                break;
            } // Just sample a few
        }

        println!("✅ TTS with all configuration options works correctly");
    }

    // Corrected Example 4: STT with engine instance (not static method)
    println!("\nExample 4: Using engine instances correctly");
    {
        // Engines should be instantiated, not used as static types
        struct MyCustomEngine;

        impl SttEngine for MyCustomEngine {
            type Conv = SttConversationBuilderImpl<
                futures::stream::Empty<Result<DummySegment, VoiceError>>,
            >;

            fn conversation(&self) -> Self::Conv {
                stt_conversation_builder(
                    |_source,
                     _vad,
                     _noise,
                     _lang,
                     _diarization,
                     _word_timestamps,
                     _timestamps_granularity,
                     _punctuation| {
                        futures::stream::empty::<Result<DummySegment, VoiceError>>()
                    },
                )
            }
        }

        let engine = MyCustomEngine; // Create instance
        let _stream = engine
            .conversation() // Use instance method
            .with_source(SpeechSource::File {
                path: "test.wav".to_string(),
                format: AudioFormat::Pcm16Khz,
            })
            .listen(|conversation| match conversation {
                Ok(conv) => Ok(conv.into_stream()),
                Err(e) => Err(e),
            })
            .await?;

        println!("✅ Engine instance usage works correctly");
    }

    // Example 5: Alternative using FluentVoice trait directly
    println!("\nExample 5: Using FluentVoice trait for unified entry");
    {
        // This is the recommended approach for library users
        let _tts = FluentVoiceImpl::tts()
            .with_speaker(
                SpeakerLine::speaker("Narrator")
                    .speak("Using the FluentVoice trait")
                    .build(),
            )
            .synthesize(|result| match result {
                Ok(conv) => Ok(conv),
                Err(e) => Err(e),
            })
            .await?;

        let _stt = FluentVoiceImpl::stt()
            .transcribe("audio.wav")
            .emit(|result| match result {
                Ok(transcript) => Ok(transcript),
                Err(e) => Err(e),
            })
            .await?;

        println!("✅ FluentVoice trait usage works correctly");
    }

    println!("\n🎉 All corrected examples compile and work!");

    Ok(())
}
