//! Fluent voice builder for elegant voice cloning API

use super::{
    Conversation, DiaSpeaker, PitchNote, VocalSpeedMod, VoiceClone, VoiceError, VoicePersona,
    VoicePool, VoiceTimber,
};
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A fluent builder for creating voice-cloned speakers
pub struct DiaVoiceBuilder {
    pool: Arc<VoicePool>,
    audio_path: PathBuf,
    name: Option<String>,
    timber: Option<VoiceTimber>,
    personas: Vec<VoicePersona>,
    speed: Option<VocalSpeedMod>,
    pitch_range: Option<(PitchNote, PitchNote)>,
}

impl DiaVoiceBuilder {
    /// Create a new voice builder from an audio file
    pub fn new(pool: Arc<VoicePool>, audio_path: impl AsRef<Path>) -> Self {
        Self {
            pool,
            audio_path: audio_path.as_ref().to_path_buf(),
            name: None,
            timber: None,
            personas: Vec::new(),
            speed: None,
            pitch_range: None,
        }
    }

    /// Set the voice name (defaults to filename if not set)
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the voice timber
    pub fn with_timber(mut self, timber: VoiceTimber) -> Self {
        self.timber = Some(timber);
        self
    }

    /// Add a personality trait
    pub fn with_persona(mut self, persona: VoicePersona) -> Self {
        self.personas.push(persona);
        self
    }

    /// Set the speed modifier
    pub fn with_speed(mut self, speed: VocalSpeedMod) -> Self {
        self.speed = Some(speed);
        self
    }

    /// Set the pitch range
    pub fn with_pitch_range(mut self, low: PitchNote, high: PitchNote) -> Self {
        self.pitch_range = Some((low, high));
        self
    }

    /// Terminal method - speak the given text
    pub fn speak(self, text: impl Into<String>) -> DiaVoiceConversationBuilder {
        let name = self.name.unwrap_or_else(|| {
            self.audio_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("voice")
                .to_string()
        });

        DiaVoiceConversationBuilder {
            pool: self.pool,
            audio_path: self.audio_path,
            voice_name: name,
            timber: self.timber,
            personas: self.personas,
            speed: self.speed,
            pitch_range: self.pitch_range,
            text: text.into(),
        }
    }
}

/// Builder for creating a conversation from a voice clone
pub struct DiaVoiceConversationBuilder {
    pool: Arc<VoicePool>,
    audio_path: PathBuf,
    voice_name: String,
    timber: Option<VoiceTimber>,
    personas: Vec<VoicePersona>,
    speed: Option<VocalSpeedMod>,
    pitch_range: Option<(PitchNote, PitchNote)>,
    text: String,
}

impl DiaVoiceConversationBuilder {
    /// Play the conversation with Result matching
    pub async fn play<F, T>(self, handler: F) -> T
    where
        F: FnOnce(Result<super::VoicePlayer, VoiceError>) -> T,
    {
        // Load voice data through the pool
        let voice_data = match self.pool.load_voice(&self.voice_name, &self.audio_path) {
            Ok(data) => data,
            Err(e) => {
                return handler(Err(VoiceError::ConfigError(format!(
                    "Failed to load voice: {e}"
                ))));
            }
        };

        // Create voice clone
        let mut voice_clone = VoiceClone::new(&self.voice_name, voice_data);

        if let Some(timber) = self.timber {
            voice_clone = voice_clone.with_timber(timber);
        }

        for persona in self.personas {
            voice_clone = voice_clone.with_persona(persona);
        }

        if let Some(speed) = self.speed {
            voice_clone = voice_clone.with_speed(speed);
        }

        if let Some((low, high)) = self.pitch_range {
            voice_clone = voice_clone.with_pitch_range(low, high);
        }

        // Create speaker
        let speaker = DiaSpeaker { voice_clone };

        // Create and play conversation
        match Conversation::new(self.text, speaker, self.pool.clone()).await {
            Ok(conversation) => conversation.internal_generate().await.pipe(handler),
            Err(e) => handler(Err(VoiceError::ConfigError(format!(
                "Failed to create conversation: {e}"
            )))),
        }
    }
}

// Helper trait for pipe operator
trait Pipe {
    fn pipe<F, T>(self, f: F) -> T
    where
        F: FnOnce(Self) -> T,
        Self: Sized,
    {
        f(self)
    }
}

impl<T> Pipe for T {}

// Type aliases for backward compatibility
pub type VoiceBuilder = DiaVoiceBuilder;
pub type VoiceConversationBuilder = DiaVoiceConversationBuilder;
