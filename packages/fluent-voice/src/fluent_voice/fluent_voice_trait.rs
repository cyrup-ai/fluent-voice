//! Unified entry point trait for TTS and STT operations.

use super::entry_points::{SttEntry, TtsEntry};
use crate::builders::{
    AudioIsolationBuilder, SoundEffectsBuilder, SpeechToSpeechBuilder, VoiceCloneBuilder,
    VoiceDiscoveryBuilder,
};
use crate::wake_word::WakeWordBuilder;

/// Unified entry point for Text-to-Speech and Speech-to-Text operations.
///
/// This trait provides the main entry points for the fluent voice API, allowing
/// users to start TTS or STT operations with a consistent interface.
///
/// # Examples
///
/// ```ignore
/// use fluent_voice::prelude::*;
///
/// // TTS usage
/// let audio = FluentVoice::tts().conversation()
///     .with_speaker(
///         Speaker::speaker("Alice")
///             .speak("Hello, world!")
///             .build()
///     )
///     .synthesize(|conversation| {
///         Ok => conversation.into_stream(),
///         Err(e) => Err(e),
///     })
///     .await?;
///
/// // STT microphone usage
/// let stream = FluentVoice::stt().conversation()
///     .with_source(SpeechSource::Microphone {
///         backend: MicBackend::Default,
///         format: AudioFormat::Pcm16Khz,
///         sample_rate: 16_000,
///     })
///     .listen(|conversation| {
///         Ok => conversation.into_stream(),
///         Err(e) => Err(e),
///     })
///     .await?;
/// ```
pub trait FluentVoice {
    /// Begin a new TTS session.
    ///
    /// This method returns an entry point that provides access to TTS conversation builders.
    ///
    /// # Returns
    ///
    /// A new TTS session instance.
    fn tts() -> TtsEntry;

    /// Begin a new STT session.
    ///
    /// This method returns an entry point that provides access to STT conversation builders.
    ///
    /// # Returns
    ///
    /// A new STT session instance.
    fn stt() -> SttEntry;

    /// Begin a new wake word detection builder.
    ///
    /// This method returns a builder that can be used to configure
    /// wake word models, confidence thresholds, and other detection
    /// parameters before starting wake word detection.
    ///
    /// # Returns
    ///
    /// A new wake word builder instance.
    fn wake_word() -> impl WakeWordBuilder;
    /// Begin a new voice discovery builder.
    ///
    /// This method returns a builder that can be used to search and
    /// filter available voices from the engine provider.
    ///
    /// # Returns
    ///
    /// A new voice discovery builder instance.
    fn voices() -> impl VoiceDiscoveryBuilder;

    /// Begin a new voice cloning builder.
    ///
    /// This method returns a builder that can be used to create
    /// custom voices from audio samples.
    ///
    /// # Returns
    ///
    /// A new voice cloning builder instance.
    fn clone_voice() -> impl VoiceCloneBuilder;

    /// Begin a new speech-to-speech conversion builder.
    ///
    /// This method returns a builder that can be used to convert
    /// speech from one voice to another while preserving characteristics.
    ///
    /// # Returns
    ///
    /// A new speech-to-speech builder instance.
    fn speech_to_speech() -> impl SpeechToSpeechBuilder;

    /// Begin a new audio isolation builder.
    ///
    /// This method returns a builder that can be used to separate
    /// voices from background audio or isolate specific audio components.
    ///
    /// # Returns
    ///
    /// A new audio isolation builder instance.
    fn audio_isolation() -> impl AudioIsolationBuilder;

    /// Begin a new sound effects generation builder.
    ///
    /// This method returns a builder that can be used to generate
    /// audio effects from text descriptions.
    ///
    /// # Returns
    ///
    /// A new sound effects builder instance.
    fn sound_effects() -> impl SoundEffectsBuilder;
}
