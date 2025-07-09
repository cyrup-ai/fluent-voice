//! Test to verify README syntax works exactly as written

use fluent_voice::prelude::*;

fn main() {
    // Test 1: Speaker::speaker syntax from README
    let _speaker = Speaker::speaker("Alice")
        .voice_id(VoiceId::new("voice-uuid"))
        .with_speed_modifier(VocalSpeedMod(0.9))
        .speak("Hello, world!")
        .build();

    println!("✅ Speaker::speaker() syntax works!");

    // Test 2: FluentVoice static methods
    let _tts = FluentVoice::tts();
    let _stt = FluentVoice::stt();

    println!("✅ FluentVoice::tts() and FluentVoice::stt() work!");
}
