//! Supported audio encodings negotiated with the engine.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}
