//! Supported audio encodings negotiated with the engine.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    /// 16-bit PCM at 16 kHz, mono.
    Pcm16Khz,
    /// 16-bit PCM at 24 kHz, mono.
    Pcm24Khz,
    /// 16-bit PCM at 48 kHz, mono.
    Pcm48Khz,
    /// MP3 44.1 kHz, 128 kbps CBR.
    Mp3Khz44_128,
    /// MP3 44.1 kHz, 192 kbps CBR.
    Mp3Khz44_192,
    /// Ogg/Opus 48 kHz, ~96 kbps VBR.
    OggOpusKhz48,

    // Extended ElevenLabs formats
    /// MP3 22.05 kHz, 32 kbps CBR.
    Mp3Khz22_32,
    /// MP3 44.1 kHz, 32 kbps CBR.
    Mp3Khz44_32,
    /// MP3 44.1 kHz, 64 kbps CBR.
    Mp3Khz44_64,
    /// MP3 44.1 kHz, 96 kbps CBR.
    Mp3Khz44_96,
    /// 16-bit PCM at 22.05 kHz, mono.
    Pcm22Khz,
    /// μ-law encoded audio at 8 kHz.
    ULaw8Khz,
}
