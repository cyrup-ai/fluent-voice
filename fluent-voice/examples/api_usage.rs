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
    let _audio = FluentVoice::tts()
        .with_speaker(
            Speaker::named("Bob")
                .with_speed_modifier(VocalSpeedMod::Slow)
                .with_pitch_range(PitchRange {
                    low: 60.0,
                    high: 150.0,
                })
                .speak("Builder-for-builders is neat!"),
        )
        .with_speaker(Speaker::named("Julie").speak("Right? Now engines take minutes, not hours."))
        .synthesize(|conversation| match conversation {
            Ok(conv) => Ok(conv.into_stream()),
            Err(err) => Err(anyhow::anyhow!(err)),
        })
        .await?;

    Ok(())
}

// These would be provided by implementation crates:
mod elevenlabs_fluent_voice {
    use fluent_voice::prelude::*;

    pub fn init() {
        // Engine initialization would register itself with FluentVoice
        // FluentVoice::register_tts_engine(ElevenLabsEngine::new());
    }

    // Example of how an engine would implement the traits
    pub struct ElevenLabsEngine;

    impl TtsEngine for ElevenLabsEngine {
        type Conv = ElevenLabsConversationBuilder;

        fn conversation(&self) -> Self::Conv {
            ElevenLabsConversationBuilder::new()
        }
    }

    pub struct ElevenLabsConversationBuilder;

    impl ElevenLabsConversationBuilder {
        fn new() -> Self {
            Self
        }
    }

    impl TtsConversationBuilder for ElevenLabsConversationBuilder {
        type Conversation = ElevenLabsConversation;

        fn with_speaker<S: Speaker>(self, _speaker: S) -> Self {
            self
        }
        fn language(self, _lang: Language) -> Self {
            self
        }
        fn synthesize<F, R>(self, _matcher: F) -> impl std::future::Future<Output = R> + Send
        where
            F: FnOnce(Result<Self::Conversation, VoiceError>) -> R + Send + 'static,
        {
            async move { unimplemented!() }
        }
    }

    pub struct ElevenLabsConversation;

    impl TtsConversation for ElevenLabsConversation {
        type AudioStream = futures::stream::Empty<i16>;
        fn into_stream(self) -> Self::AudioStream {
            futures::stream::empty()
        }
    }
}

// FluentVoice would be provided by the fluent_voice crate
// and delegate to registered engines
