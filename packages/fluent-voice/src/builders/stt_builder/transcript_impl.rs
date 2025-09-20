//! TranscriptImpl struct for file transcription

use fluent_voice_domain::transcription::TranscriptionStream;

/// Transcript collection type for file transcription.
pub struct TranscriptImpl<S: TranscriptionStream> {
    /// The transcript stream
    pub stream: S,
}
