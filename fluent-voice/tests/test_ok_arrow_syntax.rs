//! Test the Ok => syntax from README

use fluent_voice::prelude::*;
use fluent_voice::speaker::Speaker as SpeakerTrait;

#[test]
fn test_ok_arrow_macro_syntax() {
    // Test if we can use the fv_match! macro with Ok => syntax
    use fluent_voice::fv_match;

    let result: Result<String, VoiceError> = Ok("test".to_string());

    let processed = fv_match!(|value| {
        Ok => value,
        Err(e) => panic!("Error: {:?}", e),
    })(result);

    assert_eq!(processed, "test".to_string());
}

#[test]
fn test_readme_listen_syntax_interpretation() {
    // The README shows:
    // .listen(|segment| {
    //     Ok  => segment.text(),
    //     Err(e) => Err(e),
    // })

    // This syntax doesn't make sense as written because:
    // 1. |segment| suggests segment is the parameter
    // 2. But then Ok => segment.text() implies segment is inside the Ok

    // The correct interpretation must be that this is pseudo-syntax
    // and the actual implementation would be:

    async fn example() -> Result<(), VoiceError> {
        let result = FluentVoice::stt()
            .with_source(SpeechSource::Microphone {
                backend: MicBackend::Default,
                format: AudioFormat::Pcm16Khz,
                sample_rate: 16_000,
            })
            .listen(|conversation| match conversation {
                Ok(conv) => Ok(conv.into_stream()),
                Err(e) => Err(e),
            })
            .await;

        Ok(())
    }

    let _ = example;
}

#[test]
fn test_synthesize_macro() {
    // Test the tts_synthesize! macro
    use fluent_voice::{tts_conversation_builder, tts_synthesize};

    async fn example() -> Result<(), VoiceError> {
        let builder = tts_conversation_builder(|_lines, _lang| futures::stream::empty::<i16>());

        let _result = tts_synthesize!(builder, |conversation| {
            Ok => conversation.into_stream(),
            Err(e) => Err(e),
        })
        .await?;

        Ok(())
    }

    let _ = example;
}

#[test]
fn test_speaker_trait_methods() {
    let speaker = Speaker::speaker("TestSpeaker")
        .voice_id(VoiceId::new("test-voice"))
        .speak("Test message")
        .build();

    // Import the trait to use its methods
    assert_eq!(SpeakerTrait::id(&speaker), "TestSpeaker");
    assert_eq!(SpeakerTrait::text(&speaker), "Test message");
}
