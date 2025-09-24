//! AudioChunkWrapper implementation for MessageChunk trait compatibility

use cyrup_sugars::prelude::MessageChunk;

// Wrapper type to implement MessageChunk for AudioChunk (avoiding orphan rule)
#[derive(Debug, Clone)]
pub struct AudioChunkWrapper(pub fluent_voice_domain::AudioChunk);

impl MessageChunk for AudioChunkWrapper {
    fn bad_chunk(error: String) -> Self {
        AudioChunkWrapper(fluent_voice_domain::AudioChunk::with_metadata(
            Vec::new(),
            0,
            0,
            None,
            Some(format!("[ERROR] {}", error)),
            None,
        ))
    }

    fn error(&self) -> Option<&str> {
        self.0.text().and_then(|text| {
            if text.starts_with("[ERROR]") {
                Some(text[8..].trim())
            } else {
                None
            }
        })
    }

    fn is_error(&self) -> bool {
        self.0
            .text()
            .is_some_and(|text| text.starts_with("[ERROR]"))
    }
}

impl From<fluent_voice_domain::AudioChunk> for AudioChunkWrapper {
    fn from(chunk: fluent_voice_domain::AudioChunk) -> Self {
        AudioChunkWrapper(chunk)
    }
}

impl From<AudioChunkWrapper> for fluent_voice_domain::AudioChunk {
    fn from(wrapper: AudioChunkWrapper) -> Self {
        wrapper.0
    }
}
