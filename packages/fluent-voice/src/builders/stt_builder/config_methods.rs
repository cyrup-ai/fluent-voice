//! Configuration methods for SttConversationBuilderImpl

use super::conversation_builder_core::SttConversationBuilderImpl;
use fluent_voice_domain::transcription::TranscriptionStream;

impl<S> SttConversationBuilderImpl<S>
where
    S: TranscriptionStream + 'static,
{
    /// Configure engine parameters using JSON object syntax
    pub fn engine_config(
        mut self,
        config: impl Into<hashbrown::HashMap<&'static str, &'static str>>,
    ) -> Self {
        let config_map = config.into();
        for (k, v) in config_map {
            self.engine_config.insert(k.to_string(), v.to_string());
        }
        self
    }
}
