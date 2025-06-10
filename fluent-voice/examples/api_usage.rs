// Example of using the fluent_voice API
// This assumes an implementation crate (e.g., elevenlabs_fluent_voice) exists

use fluent_voice::prelude::*;

// This would come from an implementation crate
use elevenlabs_fluent_voice::init as init_eleven;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize the engine
    init_eleven();

    // Complete fluent chain with multiple speakers
    let _audio = FluentVoice::conversation()
        .with_speaker(
            Speaker::named("Bob")
                .with_speed_modifier(VocalSpeedMod::Slow)
                .with_pitch_range(PitchRange { low: 60.0, high: 150.0 })
                .speak("Builder-for-builders is neat!"),
        )
        .with_speaker(
            Speaker::named("Julie")
                .speak("Right? Now engines take minutes, not hours."),
        )
        .play(|result| {
            match result {
                Ok(audio) => Ok(audio),
                Err(err) => Err(anyhow::anyhow!(err)),
            }
        })
        .await?;

    Ok(())
}

// These would be provided by implementation crates:
mod elevenlabs_fluent_voice {
    pub fn init() {}
}

struct FluentVoice;

impl FluentVoice {
    pub fn conversation() -> impl fluent_voice::prelude::ConversationBuilder {
        unimplemented!("This would be provided by the implementation")
    }
}