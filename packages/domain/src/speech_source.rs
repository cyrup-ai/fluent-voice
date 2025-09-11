//! Audio input sources for STT.
use crate::{audio_format::AudioFormat, mic_backend::MicBackend};
use serde::{Deserialize, Serialize};

/// Audio input source for speech-to-text processing.
///
/// This enum specifies where the STT engine should read audio data from.
/// It supports both file-based input and live microphone capture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeechSource {
    /// Audio file on the local filesystem.
    ///
    /// The engine will read and decode the specified audio file
    /// for transcription. The format hint helps with proper decoding.
    File {
        /// Path to the audio file.
        path: String,
        /// Audio format of the file.
        format: AudioFormat,
    },

    /// Live microphone capture.
    ///
    /// The engine will capture audio from a microphone device
    /// in real-time for transcription.
    Microphone {
        /// Which microphone device to use.
        backend: MicBackend,
        /// Audio format for capture.
        format: AudioFormat,
        /// Sample rate in Hz.
        sample_rate: u32,
    },

    /// Audio data in memory buffer.
    ///
    /// The engine will process audio data directly from memory
    /// for transcription.
    Memory {
        /// Audio data buffer.
        data: Vec<u8>,
        /// Audio format of the data.
        format: AudioFormat,
        /// Sample rate in Hz.
        sample_rate: u32,
    },
}
