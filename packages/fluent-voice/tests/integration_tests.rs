//! Integration tests verifying the fluent API works as designed

use fluent_voice::prelude::*;
use futures_util::StreamExt;

#[tokio::test]
async fn test_complete_tts_flow() {
    // Create a test TTS engine
    struct TestTtsEngine;
    impl TtsEngine for TestTtsEngine {
        type Conv = TtsConversationBuilderImpl<futures::stream::Empty<i16>>;
        fn conversation(&self) -> Self::Conv {
            tts_conversation_builder(|lines, _lang| {
                // In a real implementation, this would generate audio from the lines
                println!("TTS Engine received {} speaker lines", lines.len());
                for line in lines {
                    println!("  {} says: {}", line.id, line.text);
                }
                futures::stream::empty::<i16>()
            })
        }
    }

    let engine = TestTtsEngine;

    // Test the complete fluent API
    let mut audio_stream = engine
        .conversation()
        .with_speaker(
            Speaker::speaker("Alice")
                .voice_id(VoiceId::new("alice-voice"))
                .with_speed_modifier(VocalSpeedMod(0.9))
                .speak("Hello from Alice!")
                .build(),
        )
        .with_speaker(
            Speaker::speaker("Bob")
                .voice_id(VoiceId::new("bob-voice"))
                .with_speed_modifier(VocalSpeedMod(1.1))
                .speak("Hi Alice, this is Bob!")
                .build(),
        )
        .language(Language("en-US"))
        .synthesize(|result| match result {
            Ok(conversation) => conversation.into_stream(),
            Err(e) => panic!("TTS synthesis failed: {:?}", e),
        })
        .await;

    // Verify we can iterate the stream (even though it's empty in this test)
    let mut count = 0;
    while let Some(_sample) = audio_stream.next().await {
        count += 1;
    }
    assert_eq!(count, 0); // Empty stream for test
}

#[tokio::test]
async fn test_complete_stt_flow() {
    // Test the STT fluent API
    let transcript_stream = FluentVoice::stt()
        .with_source(SpeechSource::Microphone {
            backend: MicBackend::Default,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        })
        .vad_mode(VadMode::Accurate)
        .language_hint(Language("en-US"))
        .diarization(Diarization::On)
        .word_timestamps(WordTimestamps::On)
        .punctuation(Punctuation::On)
        .listen(|result| match result {
            Ok(conversation) => Ok(conversation.into_stream()),
            Err(e) => Err(e),
        })
        .await;

    // The default implementation returns an empty stream
    futures::pin_mut!(transcript_stream);
    let mut count = 0;
    while let Some(_segment) = transcript_stream.next().await {
        count += 1;
    }
    assert_eq!(count, 0); // Empty stream for default impl
}

#[tokio::test]
async fn test_stt_collect_method() {
    // Test the collect() method on SttConversation
    let transcript = FluentVoice::stt()
        .with_source(SpeechSource::File {
            path: "test.wav".to_string(),
            format: AudioFormat::Pcm16Khz,
        })
        .listen(|result| match result {
            Ok(conversation) => Ok(conversation),
            Err(e) => Err(e),
        })
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // Default implementation returns empty string
    assert_eq!(transcript, "");
}

#[test]
fn test_builder_chaining() {
    // Test that all builder methods can be chained
    let _builder = FluentVoice::stt()
        .with_source(SpeechSource::File {
            path: "test.wav".to_string(),
            format: AudioFormat::Pcm16Khz,
        })
        .vad_mode(VadMode::Accurate)
        .noise_reduction(NoiseReduction::High)
        .language_hint(Language("en-US"))
        .diarization(Diarization::On)
        .word_timestamps(WordTimestamps::On)
        .timestamps_granularity(TimestampsGranularity::Word)
        .punctuation(Punctuation::On);

    // Verify it compiles
}

#[test]
fn test_microphone_builder() {
    // Test microphone-specific builder
    async fn test() -> Result<(), VoiceError> {
        let _stream = FluentVoice::stt()
            .with_microphone("default")
            .vad_mode(VadMode::Accurate)
            .language_hint(Language("es-ES"))
            .listen(|result| match result {
                Ok(conv) => conv.into_stream(),
                Err(_) => futures::stream::empty(),
            });
        Ok(())
    }

    let _ = test;
}

#[test]
fn test_transcription_builder() {
    // Test file transcription builder
    async fn test() -> Result<(), VoiceError> {
        let _transcript = FluentVoice::stt()
            .transcribe("audio.mp3")
            .language_hint(Language("fr-FR"))
            .with_progress("Transcribing {percent}%...")
            .collect()
            .await?;
        Ok(())
    }

    let _ = test;
}

#[test]
fn test_all_tts_settings() {
    // Test all TTS configuration options
    let _builder = FluentVoice::tts()
        .model(ModelId::TurboV2_5)
        .stability(Stability::default())
        .similarity(Similarity::default())
        .speaker_boost(SpeakerBoost::default())
        .style_exaggeration(StyleExaggeration::default())
        .language(Language("de-DE"));

    // Verify it compiles
}

#[test]
fn test_speaker_builder_all_options() {
    // Test all speaker builder options
    let speaker = Speaker::speaker("TestSpeaker")
        .voice_id(VoiceId::new("test-voice-123"))
        .language(Language("ja-JP"))
        .with_speed_modifier(VocalSpeedMod(1.2))
        .with_pitch_range(PitchRange::new(0.8, 1.2))
        .speak("Testing all options")
        .build();

    // Use the Speaker trait to access methods
    use fluent_voice::speaker::Speaker as SpeakerTrait;
    assert_eq!(SpeakerTrait::id(&speaker), "TestSpeaker");
    assert_eq!(SpeakerTrait::text(&speaker), "Testing all options");
}

#[test]
fn test_value_types() {
    // Test that all value types can be constructed
    let _ = Language("en-GB");
    let _ = VoiceId::new("voice-123");
    let _ = ModelId::MultilingualV2;
    let _ = VocalSpeedMod(1.5);
    let _ = AudioFormat::Pcm16Khz;
    let _ = MicBackend::Device("my-mic");
    let _ = VadMode::Fast;
    let _ = NoiseReduction::Low;
    let _ = Diarization::Off;
    let _ = WordTimestamps::On;
    let _ = Punctuation::Off;
    let _ = TimestampsGranularity::Word;
}

// Verify the macros work correctly
#[cfg(feature = "macros")]
mod macro_tests {
    use super::*;

    #[test]
    fn test_tts_macro() {
        let _builder = tts!();
    }

    #[test]
    fn test_stt_macro() {
        let _builder = stt!();
    }
}
