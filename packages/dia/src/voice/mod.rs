//! Voice synthesis module - trait-based API for Dia TTS

pub mod async_voice;
pub mod cli;
pub mod clone;
pub mod codec;
pub mod conversation;
pub mod dia_speaker;
pub mod macros;
pub mod persona;
pub mod pitch;
pub mod pool;
pub mod speaker;
pub mod speed;
pub mod timber;
pub mod voice_builder;
pub mod voice_player;

pub use async_voice::{AsyncVoice, VoiceError};
pub use clone::{SegmentConfig, VoiceClone, VoiceCloneBuilder};
pub use codec::{VoiceCodec, VoiceData};
pub use conversation::Conversation;
pub use dia_speaker::{DiaSpeaker, DiaSpeakerBuilder};
pub use persona::VoicePersona;
pub use pitch::{Note, Octave, PitchNote};
pub use pool::{VoicePool, global_pool, init_global_pool};
pub use speaker::{Speaker, SpeakerBuilder};
pub use speed::VocalSpeedMod;
pub use timber::VoiceTimber;
pub use voice_builder::{VoiceBuilder, VoiceConversationBuilder};
pub use voice_player::{AudioChunk, VoicePlayer};
