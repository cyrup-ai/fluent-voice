
pub mod vocal_speed_mod;
pub mod voice_timber;
pub mod pitch_range;
pub mod audio_stream;
pub mod voice_error;
pub mod speaker;
pub mod speaker_builder;
pub mod conversation_builder;
pub mod voice_player;
pub mod engine;

mod internal_macro;              // hidden macro; NOT re-exported

/* Public prelude re-exporting just the traits & enums */
pub mod prelude {
    pub use crate::{
        audio_stream::AudioStream,
        conversation_builder::ConversationBuilder,
        engine::Engine,
        pitch_range::PitchRange,
        speaker::Speaker,
        speaker_builder::{SpeakerBuilder, SpeakerExt},
        vocal_speed_mod::VocalSpeedMod,
        voice_error::VoiceError,
        voice_player::VoicePlayer,
        voice_timber::VoiceTimber,
    };
}