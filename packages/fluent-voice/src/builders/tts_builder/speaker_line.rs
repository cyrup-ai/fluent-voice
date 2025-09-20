//! SpeakerLine struct and Speaker trait implementation

use crate::speaker_builder::SpeakerBuilder;

/// High-performance speaker line with zero-allocation design
#[derive(Clone, Debug)]
pub struct SpeakerLine {
    pub id: String,
    pub text: String,
    pub voice_id: Option<fluent_voice_domain::VoiceId>,
    pub language: Option<fluent_voice_domain::Language>,
    pub speed_modifier: Option<fluent_voice_domain::VocalSpeedMod>,
    pub pitch_range: Option<fluent_voice_domain::PitchRange>,
    pub metadata: hashbrown::HashMap<String, String>,
    pub vocal_settings: hashbrown::HashMap<String, String>,
}

impl SpeakerLine {
    #[inline]
    pub fn new(
        name: impl Into<String>,
    ) -> crate::builders::tts_builder::speaker_line_builder::SpeakerLineBuilder {
        crate::builders::tts_builder::speaker_line_builder::SpeakerLineBuilder::speaker(name)
    }

    #[inline]
    pub fn speaker(
        name: impl Into<String>,
    ) -> crate::builders::tts_builder::speaker_line_builder::SpeakerLineBuilder {
        crate::builders::tts_builder::speaker_line_builder::SpeakerLineBuilder::speaker(name)
    }
}

impl crate::speaker::Speaker for SpeakerLine {
    #[inline]
    fn id(&self) -> &str {
        &self.id
    }

    #[inline]
    fn text(&self) -> &str {
        &self.text
    }

    #[inline]
    fn voice_id(&self) -> Option<&fluent_voice_domain::VoiceId> {
        self.voice_id.as_ref()
    }

    #[inline]
    fn language(&self) -> Option<&fluent_voice_domain::Language> {
        self.language.as_ref()
    }

    #[inline]
    fn speed_modifier(&self) -> Option<fluent_voice_domain::VocalSpeedMod> {
        self.speed_modifier
    }

    #[inline]
    fn pitch_range(&self) -> Option<&fluent_voice_domain::PitchRange> {
        self.pitch_range.as_ref()
    }
}
