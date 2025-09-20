//! Additional configuration methods for TtsConversationBuilderImpl

use super::builder_core::TtsConversationBuilderImpl;
use futures::Stream;

impl<AudioStream> TtsConversationBuilderImpl<AudioStream>
where
    AudioStream: Stream<Item = fluent_voice_domain::AudioChunk> + Send + Unpin + 'static,
{
    /// Configure timestamp granularity using domain types
    pub fn timestamp_granularity(
        mut self,
        granularity: fluent_voice_domain::timestamps::TimestampsGranularity,
    ) -> Self {
        self.metadata.insert(
            "timestamp_granularity".to_string(),
            serde_json::to_string(&granularity).unwrap_or_default(),
        );
        self
    }

    /// Configure word timestamp inclusion using domain types
    pub fn word_timestamps(
        mut self,
        enabled: fluent_voice_domain::timestamps::WordTimestamps,
    ) -> Self {
        self.metadata.insert(
            "word_timestamps".to_string(),
            serde_json::to_string(&enabled).unwrap_or_default(),
        );
        self
    }

    /// Configure speaker diarization using domain types
    pub fn diarization(mut self, enabled: fluent_voice_domain::timestamps::Diarization) -> Self {
        self.metadata.insert(
            "diarization".to_string(),
            serde_json::to_string(&enabled).unwrap_or_default(),
        );
        self
    }

    /// Configure punctuation insertion using domain types
    pub fn punctuation(mut self, enabled: fluent_voice_domain::timestamps::Punctuation) -> Self {
        self.metadata.insert(
            "punctuation".to_string(),
            serde_json::to_string(&enabled).unwrap_or_default(),
        );
        self
    }
}
