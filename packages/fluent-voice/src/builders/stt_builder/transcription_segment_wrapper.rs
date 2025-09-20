//! TranscriptionSegmentWrapper implementation for MessageChunk trait compatibility

use cyrup_sugars::prelude::MessageChunk;
use fluent_voice_domain::TranscriptionSegment;

// Wrapper type to implement MessageChunk for TranscriptionSegmentImpl (avoiding orphan rule)
#[derive(Debug, Clone)]
pub struct TranscriptionSegmentWrapper(pub fluent_voice_domain::TranscriptionSegmentImpl);

impl MessageChunk for TranscriptionSegmentWrapper {
    fn bad_chunk(error: String) -> Self {
        TranscriptionSegmentWrapper(fluent_voice_domain::TranscriptionSegmentImpl::new(
            format!("[ERROR] {}", error),
            0,
            0,
            None,
        ))
    }

    fn error(&self) -> Option<&str> {
        if self.0.text().starts_with("[ERROR]") {
            Some(&self.0.text()[8..].trim())
        } else {
            None
        }
    }

    fn is_error(&self) -> bool {
        self.0.text().starts_with("[ERROR]")
    }
}

impl From<fluent_voice_domain::TranscriptionSegmentImpl> for TranscriptionSegmentWrapper {
    fn from(segment: fluent_voice_domain::TranscriptionSegmentImpl) -> Self {
        TranscriptionSegmentWrapper(segment)
    }
}

impl From<TranscriptionSegmentWrapper> for fluent_voice_domain::TranscriptionSegmentImpl {
    fn from(wrapper: TranscriptionSegmentWrapper) -> Self {
        wrapper.0
    }
}
