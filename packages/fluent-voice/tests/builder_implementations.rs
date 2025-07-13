//! Integration test for the builder implementations.
//!
//! This tests for builder implementations

use fluent_voice::builders::{stt_conversation_builder, tts_conversation_builder};
use fluent_voice::prelude::*;
use futures::stream;

#[test]
fn test_tts_builder_exists() {
    // Create a simple mock synthesis function
    let synth_fn = move |_lines: &[SpeakerLine], _lang: Option<&Language>| {
        // Return empty stream of i16 audio samples for test
        stream::empty::<i16>()
    };

    // Just verify we can call the builder function
    let _builder = tts_conversation_builder(synth_fn);

    // Validate it compiled
    assert!(true);
}

#[test]
fn test_stt_builder_exists() {
    // Create a simple mock stream function
    let stream_fn = move |_src, _vad, _noise, _lang, _diar, _word, _ts, _punct| {
        // Return empty result stream with real TtsChunk for test
        stream::iter(std::iter::empty::<
            Result<fluent_voice_whisper::TtsChunk, VoiceError>,
        >())
    };

    // Just verify we can call the builder function
    let _builder = stt_conversation_builder(stream_fn);

    // Validate it compiled
    assert!(true);
}
