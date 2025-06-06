//! fluent_voice/src/engine.rs
//! --------------------------
//! Engine trait definition

use crate::{
    conversation_builder::ConversationBuilder,
    speaker_builder::SpeakerBuilder,
};

/* Global hook each vendor crate provides */
pub trait Engine: Send + Sync {
    type SpeakerB: SpeakerBuilder;
    type Conversation:
        ConversationBuilder<PendingSpeaker = <Self::SpeakerB as SpeakerBuilder>::Speaker>;

    fn conversation(&self) -> Self::Conversation;
}