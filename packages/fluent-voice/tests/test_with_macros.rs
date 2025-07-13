//! Test using the internal macros to get closer to README syntax

use fluent_voice::prelude::*;
use fluent_voice::{fv_match, stt_listen, tts_synthesize};
use futures_util::StreamExt;

#[tokio::test]
async fn test_stt_with_macro() -> Result<(), VoiceError> {
    // Using the fv_match! macro
    let conversation = FluentVoice::stt()
        .with_source(SpeechSource::Microphone {
            backend: MicBackend::Default,
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        })
        .listen(fv_match!(|conv| {
            Ok => conv,
            Err(e) => panic!("Error: {:?}", e),
        }))
        .await;

    let transcript = conversation.collect().await?;
    Ok(())
}

#[tokio::test]
async fn test_using_stt_listen_macro() -> Result<(), VoiceError> {
    let builder = FluentVoice::stt().with_source(SpeechSource::Microphone {
        backend: MicBackend::Default,
        format: AudioFormat::Pcm16Khz,
        sample_rate: 16_000,
    });

    // Use the stt_listen! macro
    let conversation = stt_listen!(builder, |conv| {
        Ok => conv,
        Err(e) => panic!("Error: {:?}", e),
    })
    .await;

    Ok(())
}
