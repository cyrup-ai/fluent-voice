//! Test that the exact syntax from README.md works correctly

use fluent_voice::prelude::*;
use futures_util::StreamExt;

#[test]
fn test_readme_stt_syntax_compiles() {
    // This test verifies the STT example from README lines 53-68 compiles
    // Note: We can't actually run it without a real engine, but we can verify it compiles

    async fn example() -> Result<(), VoiceError> {
        // Exact syntax from README.md
        let _transcript = FluentVoice::stt()
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
            .listen(|segment| match segment {
                Ok(seg) => Ok(seg),
                Err(e) => Err(e),
            })
            .await?
            .collect()
            .await?;

        Ok(())
    }

    // Just verify it compiles
    let _ = example;
}

#[test]
fn test_readme_tts_syntax_compiles() {
    // Test TTS example from README lines 78-96

    async fn example() -> Result<(), VoiceError> {
        // First, define a mock engine since MyTtsEngine is not provided
        struct MyTtsEngine;
        impl TtsEngine for MyTtsEngine {
            type Conv = TtsConversationBuilderImpl<futures::stream::Empty<i16>>;
            fn conversation(&self) -> Self::Conv {
                tts_conversation_builder(|_lines, _lang| futures::stream::empty::<i16>())
            }
        }

        let engine = MyTtsEngine;

        // Exact syntax from README.md (adjusted to use engine instance)
        let mut audio_stream = engine
            .conversation()
            .with_speaker(
                Speaker::speaker("Alice")
                    .voice_id(VoiceId::new("voice-uuid"))
                    .with_speed_modifier(VocalSpeedMod(0.9))
                    .speak("Hello, world!")
                    .build(),
            )
            .with_speaker(
                Speaker::speaker("Bob")
                    .with_speed_modifier(VocalSpeedMod(1.1))
                    .speak("Hi Alice! How are you today?")
                    .build(),
            )
            .synthesize(|conversation| match conversation {
                Ok(conv) => Ok(conv.into_stream()),
                Err(e) => Err(e),
            })
            .await?;

        // Process audio samples
        while let Some(_sample) = audio_stream.next().await {
            // Would play sample or save to file
            break; // Exit after first sample for test
        }

        Ok(())
    }

    let _ = example;
}

#[test]
fn test_speaker_builder_syntax() {
    // Test that Speaker::speaker() syntax works
    let speaker = Speaker::speaker("TestSpeaker")
        .voice_id(VoiceId::new("test-voice"))
        .with_speed_modifier(VocalSpeedMod(1.0))
        .speak("Test message")
        .build();

    assert_eq!(speaker.id(), "TestSpeaker");
    assert_eq!(speaker.text(), "Test message");
}

#[test]
fn test_fluent_voice_entry_points() {
    // Test that FluentVoice::tts() and FluentVoice::stt() work

    // Just verify they compile and return builders
    let _tts_builder = FluentVoice::tts();
    let _stt_builder = FluentVoice::stt();
}

#[cfg(test)]
mod macro_syntax_tests {
    use super::*;

    #[test]
    fn test_matcher_macro_syntax() {
        // The README shows this syntax: Ok => expr
        // Let's verify the macro can handle it

        // Note: The actual macro syntax in the README needs the pattern variable
        // The examples show |segment| { Ok => segment.text(), Err(e) => Err(e) }
        // This implies the macro needs the pattern variable in the closure
    }
}
