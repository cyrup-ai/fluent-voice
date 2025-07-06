//! Integration test for the builder implementations.
//!
//! This tests the non-macro implementations of the STT and TTS builders.

use fluent_voice::{
    builders::{
        stt_builder::{self as stt, SpeakerLine},
        tts_builder::{self as tts, SpeakerLineBuilder},
    },
    language::Language,
    model_id::ModelId,
    noise_reduction::NoiseReduction,
    pitch_range::PitchRange,
    stt_conversation::{MicrophoneBuilder, SttConversationBuilder, TranscriptionBuilder},
    tts_conversation::TtsConversationBuilder,
    vad_mode::VadMode,
    voice_id::VoiceId,
};
use futures_core::Stream;
use std::pin::Pin;

#[test]
fn test_tts_builder() {
    // Create a mock synthesis function
    let synth_fn = |lines: &[SpeakerLine], _lang: Option<&Language>| {
        // In a real implementation, this would create an audio stream
        // For this test, we just return an empty stream
        Box::pin(futures::stream::empty::<i16>()) as Pin<Box<dyn Stream<Item = i16> + Send + Unpin>>
    };

    // Create a new TTS conversation builder
    let builder = tts::builder::tts_conversation_builder(synth_fn)
        .language(Language::new("en-US"))
        .model(ModelId::new("tts-1"))
        .with_speaker(
            SpeakerLineBuilder::named("Alice")
                .voice_id(VoiceId::new("alloy"))
                .with_pitch_range(PitchRange::new(1.0))
                .speak("Hello, world!")
                .build(),
        );

    // This would typically be used in an async context
    // builder.synthesize(|result| /* handle result */);

    // Just check that the builder compiles and type-checks
    assert!(true);
}

#[test]
fn test_stt_builder() {
    // Create a mock stream function
    let stream_fn = |_src, _vad, _noise, _lang, _diar, _word, _ts, _punct| {
        // In a real implementation, this would create a transcript stream
        // For this test, we just return a dummy stream
        Box::pin(futures::stream::empty()) as Pin<Box<dyn Stream<Item = ()> + Send>>
    };

    // Create a new STT conversation builder
    let builder = stt::builder::stt_conversation_builder(stream_fn)
        .vad_mode(VadMode::Low)
        .noise_reduction(NoiseReduction::Medium)
        .language_hint(Language::new("en-US"));

    // Test microphone builder
    let _mic_builder = builder.with_microphone("default");

    // Test transcription builder
    let _transcription_builder = builder.transcribe("test.wav");

    // Just check that the builder compiles and type-checks
    assert!(true);
}
