//! Modularized Kyutai engine implementation
//!
//! This module provides a decomposed, maintainable implementation of the FluentVoice
//! trait for Kyutai's Moshi model, organized into logical separation of concerns.

pub mod audio_builders;
pub mod core_engine;
pub mod sessions;
pub mod stt_builders;
pub mod tts_builders;
pub mod voice_builders;

// Re-export public types for compatibility
pub use audio_builders::{KyutaiAudioIsolationBuilder, KyutaiSoundEffectsBuilder};
pub use core_engine::KyutaiEngine;
pub use sessions::{
    KyutaiAudioIsolationSession, KyutaiSoundEffectsSession, KyutaiSpeechToSpeechSession,
    KyutaiSttConversation, KyutaiTranscriptSegment,
};
pub use stt_builders::{
    KyutaiMicrophoneBuilder, KyutaiSttConversationBuilder, KyutaiSttPostChunkBuilder,
    KyutaiTranscriptionBuilder,
};
pub use tts_builders::{
    KyutaiSpeakerLine, KyutaiTtsConversation, KyutaiTtsConversationBuilder,
    KyutaiTtsConversationChunkBuilder,
};
pub use voice_builders::{
    KyutaiSpeechToSpeechBuilder, KyutaiVoiceCloneBuilder, KyutaiVoiceDiscoveryBuilder,
};

// Helper function to convert f32 samples to PCM16 bytes
pub fn f32_to_pcm16_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let pcm_sample = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        bytes.extend_from_slice(&pcm_sample.to_le_bytes());
    }
    bytes
}
