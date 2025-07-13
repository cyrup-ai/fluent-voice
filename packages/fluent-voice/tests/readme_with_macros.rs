//! Test README examples using the macros

use fluent_voice::prelude::*;
use fluent_voice::{fv_match, stt_listen, tts_synthesize};
use futures_util::StreamExt;

#[tokio::test]
async fn test_readme_stt_with_macro() -> Result<(), VoiceError> {
    // README example adapted to use macro
    let _transcript = stt_listen!(
        FluentVoice::stt()
            .with_source(SpeechSource::Microphone {
                backend: MicBackend::Default,
                format: AudioFormat::Pcm16Khz,
                sample_rate: 16_000,
            })
            .vad_mode(VadMode::Accurate)
            .language_hint(Language("en-US"))
            .diarization(Diarization::On)
            .word_timestamps(WordTimestamps::On)
            .punctuation(Punctuation::On),
        |segment| {
            Ok => segment,  // The macro handles the transformation
            Err(e) => panic!("{:?}", e),
        }
    )
    .await
    .collect()
    .await?;

    Ok(())
}

#[tokio::test]
async fn test_readme_tts_with_macro() -> Result<(), VoiceError> {
    struct MyTtsEngine;
    impl TtsEngine for MyTtsEngine {
        type Conv = TtsConversationBuilderImpl<futures::stream::Empty<i16>>;
        fn conversation(&self) -> Self::Conv {
            tts_conversation_builder(|_lines, _lang| futures::stream::empty::<i16>())
        }
    }

    let engine = MyTtsEngine;

    let mut audio_stream = tts_synthesize!(
        engine.conversation()
            .with_speaker(
                Speaker::speaker("Alice")
                    .voice_id(VoiceId::new("voice-uuid"))
                    .with_speed_modifier(VocalSpeedMod(0.9))
                    .speak("Hello, world!")
                    .build()
            )
            .with_speaker(
                Speaker::speaker("Bob")
                    .with_speed_modifier(VocalSpeedMod(1.1))
                    .speak("Hi Alice! How are you today?")
                    .build()
            ),
        |conversation| {
            Ok => conversation.into_stream(),
            Err(e) => panic!("{:?}", e),
        }
    )
    .await;

    while let Some(_sample) = audio_stream.next().await {
        break;
    }

    Ok(())
}
