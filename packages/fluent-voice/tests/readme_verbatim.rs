//! Test the EXACT syntax from README.md verbatim

use fluent_voice::prelude::*;
use futures_util::StreamExt;

#[tokio::test]
async fn test_readme_stt_verbatim() -> Result<(), VoiceError> {
    // Exact syntax from README lines 53-68
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
            Ok(seg) => Ok(seg.text()), // streaming chunks
            Err(e) => Err(e),
        })
        .collect(); // transcript is now the end-state string

    Ok(())
}

#[tokio::test]
async fn test_readme_tts_verbatim() -> Result<(), VoiceError> {
    // Create MyTtsEngine as shown in README
    struct MyTtsEngine;
    impl TtsEngine for MyTtsEngine {
        type Conv = TtsConversationBuilderImpl<futures::stream::Empty<i16>>;
        fn conversation(&self) -> Self::Conv {
            tts_conversation_builder(|_lines, _lang| futures::stream::empty::<i16>())
        }
    }

    // Exact syntax from README lines 78-96
    let mut audio_stream = MyTtsEngine::conversation()
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
            Ok(conv) => Ok(conv.into_stream()), // Returns audio stream
            Err(e) => Err(e),
        })
        .await?; // Single await point

    // Process audio samples
    while let Some(sample) = audio_stream.next().await {
        // Play sample or save to file
        break;
    }

    Ok(())
}
