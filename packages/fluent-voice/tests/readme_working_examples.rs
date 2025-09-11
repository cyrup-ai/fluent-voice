//! Working examples that demonstrate the actual API syntax
//! These examples show how the API actually works vs the pseudo-code in README

use fluent_voice::prelude::*;
use futures_util::StreamExt;

#[tokio::test]
async fn test_actual_stt_syntax() -> Result<(), VoiceError> {
    // The README shows this pseudo-code:
    // .listen(|segment| {
    //     Ok  => segment.text(),
    //     Err(e) => Err(e),
    // })

    // But the actual API works like this:
    let transcript = FluentVoice::stt()
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
        .listen(|result| {
            // The matcher receives a Result<Conversation, Error>
            match result {
                Ok(conversation) => {
                    // We need to return the actual value we want
                    // In this case, the conversation itself
                    Ok(conversation)
                }
                Err(e) => Err(e),
            }
        })
        .await? // This unwraps the Result
        .collect() // Now we can call collect() on the conversation
        .await?;

    println!("Transcript: {}", transcript);
    Ok(())
}

#[tokio::test]
async fn test_actual_tts_syntax() -> Result<(), VoiceError> {
    // Create a test engine
    struct TestEngine;
    impl TtsEngine for TestEngine {
        type Conv = TtsConversationBuilderImpl<futures::stream::Iter<std::vec::IntoIter<i16>>>;
        fn conversation(&self) -> Self::Conv {
            tts_conversation_builder(|_lines, _lang| {
                // Return a stream with some test samples
                futures::stream::iter(vec![0i16; 100])
            })
        }
    }

    let engine = TestEngine;

    // The README shows:
    // .synthesize(|conversation| {
    //     Ok  => conversation.into_stream(),
    //     Err(e) => Err(e),
    // })

    // But the actual API expects consistent types from both arms
    let mut audio_stream = engine
        .conversation()
        .with_speaker(
            Speaker::speaker("Alice")
                .voice_id(VoiceId::new("alice-voice"))
                .with_speed_modifier(VocalSpeedMod(0.9))
                .speak("Hello, world!")
                .build(),
        )
        .synthesize(|result| match result {
            Ok(conversation) => conversation.into_stream(),
            Err(e) => panic!("Synthesis failed: {:?}", e),
        })
        .await;

    // Process samples
    let mut count = 0;
    while let Some(sample) = audio_stream.next().await {
        count += 1;
        if count > 10 {
            break;
        } // Just process a few for testing
    }

    assert!(count > 0);
    Ok(())
}

#[test]
fn test_readme_vs_actual_differences() {
    println!("Key differences between README pseudo-code and actual API:");
    println!();
    println!("1. STT listen() method:");
    println!("   README shows: |segment| {{ Ok => segment.text(), Err(e) => Err(e) }}");
    println!("   Actual API:   |result| match result {{ Ok(conv) => Ok(conv), Err(e) => Err(e) }}");
    println!();
    println!("2. The 'Ok =>' syntax is pseudo-code, not valid Rust");
    println!("   You need proper match expressions");
    println!();
    println!("3. The closure parameter is the Result, not the unwrapped value");
    println!("   So |segment| is misleading - it's actually |result|");
    println!();
    println!("4. TTS synthesize() requires both arms to return the same type");
    println!("   So you can't return a stream from Ok and Err from Err branch");
}

#[tokio::test]
async fn test_using_macros_for_cleaner_syntax() -> Result<(), VoiceError> {
    // The crate provides macros to simplify the syntax
    use fluent_voice::{stt_listen, tts_synthesize};

    // With macros, we can get closer to the README syntax:
    let builder = FluentVoice::stt().with_source(SpeechSource::Microphone {
        backend: MicBackend::Default,
        format: AudioFormat::Pcm16Khz,
        sample_rate: 16_000,
    });

    let conversation = stt_listen!(builder, |conv| {
        Ok => conv,
        Err(e) => return Err(e),
    })
    .await?;

    // Note: The macro still requires valid Rust in the arms

    Ok(())
}
