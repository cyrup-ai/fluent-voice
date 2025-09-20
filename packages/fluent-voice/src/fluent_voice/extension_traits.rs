//! Extension trait implementations for FluentVoiceImpl.

use super::default_implementation::FluentVoiceImpl;
use super::fluent_voice_trait::FluentVoice;
use crate::stt_conversation::{SttConversationBuilder, SttConversationExt};
use crate::tts_conversation::{TtsConversationBuilder, TtsConversationExt};
use crate::wake_word::{WakeWordBuilder, WakeWordConversationExt};

/// Implementation of TtsConversationExt for FluentVoiceImpl
impl TtsConversationExt for FluentVoiceImpl {
    fn builder() -> impl TtsConversationBuilder {
        <FluentVoiceImpl as FluentVoice>::tts().conversation()
    }
}

/// Implementation of SttConversationExt for FluentVoiceImpl
impl SttConversationExt for FluentVoiceImpl {
    fn builder() -> impl SttConversationBuilder {
        <FluentVoiceImpl as FluentVoice>::stt().conversation()
    }
}

/// Implementation of WakeWordConversationExt for FluentVoiceImpl
impl WakeWordConversationExt for FluentVoiceImpl {
    fn builder() -> impl WakeWordBuilder {
        <FluentVoiceImpl as FluentVoice>::wake_word()
    }
}
