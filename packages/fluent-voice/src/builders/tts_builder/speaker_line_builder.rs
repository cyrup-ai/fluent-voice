//! SpeakerLineBuilder implementation

use super::speaker_line::SpeakerLine;

/// Zero-allocation speaker builder with optimized construction
#[derive(Clone, Debug)]
pub struct SpeakerLineBuilder {
    id: String,
    text: String,
    voice_id: Option<fluent_voice_domain::VoiceId>,
    language: Option<fluent_voice_domain::Language>,
    speed_modifier: Option<fluent_voice_domain::VocalSpeedMod>,
    pitch_range: Option<fluent_voice_domain::PitchRange>,
    metadata: hashbrown::HashMap<String, String>,
    vocal_settings: hashbrown::HashMap<String, String>,
}

impl crate::speaker_builder::SpeakerBuilder for SpeakerLineBuilder {
    type Output = SpeakerLine;

    #[inline]
    fn speaker(name: impl Into<String>) -> Self {
        SpeakerLineBuilder {
            id: name.into(),
            text: String::new(),
            voice_id: None,
            language: None,
            speed_modifier: None,
            pitch_range: None,
            metadata: hashbrown::HashMap::new(),
            vocal_settings: hashbrown::HashMap::new(),
        }
    }

    #[inline]
    fn voice_id(mut self, id: fluent_voice_domain::VoiceId) -> Self {
        self.voice_id = Some(id);
        self
    }

    #[inline]
    fn language(mut self, lang: fluent_voice_domain::Language) -> Self {
        self.language = Some(lang);
        self
    }

    #[inline]
    fn with_prelude(mut self, prelude: impl Into<String>) -> Self {
        let prelude_text = prelude.into();
        if !self.text.is_empty() {
            self.text.push(' ');
        }
        self.text.push_str(&prelude_text);
        self
    }

    #[inline]
    fn add_line(mut self, line: impl Into<String>) -> Self {
        let line_text = line.into();
        if !self.text.is_empty() {
            self.text.push(' ');
        }
        self.text.push_str(&line_text);
        self
    }

    #[inline]
    fn with_voice(mut self, voice: impl Into<String>) -> Self {
        let voice_str = voice.into();
        self.voice_id = Some(fluent_voice_domain::VoiceId::new(voice_str));
        self
    }

    #[inline]
    fn with_speed(mut self, speed: f32) -> Self {
        self.speed_modifier = Some(fluent_voice_domain::VocalSpeedMod(speed));
        self
    }

    #[inline]
    fn with_speed_modifier(mut self, m: fluent_voice_domain::VocalSpeedMod) -> Self {
        self.speed_modifier = Some(m);
        self
    }

    #[inline]
    fn with_pitch_range(mut self, range: fluent_voice_domain::PitchRange) -> Self {
        self.pitch_range = Some(range);
        self
    }

    #[inline]
    fn speak(mut self, text: impl Into<String>) -> Self {
        self.text = text.into();
        self
    }

    #[inline]
    fn build(self) -> Self::Output {
        SpeakerLine {
            id: self.id,
            text: self.text,
            voice_id: self.voice_id,
            language: self.language,
            speed_modifier: self.speed_modifier,
            pitch_range: self.pitch_range,
            metadata: self.metadata,
            vocal_settings: self.vocal_settings,
        }
    }
}

impl SpeakerLineBuilder {
    /// Zero-allocation metadata configuration
    #[inline]
    pub fn metadata(
        mut self,
        config: impl Into<hashbrown::HashMap<&'static str, &'static str>>,
    ) -> Self {
        let config_map = config.into();
        for (k, v) in config_map {
            self.metadata.insert(k.to_string(), v.to_string());
        }
        self
    }

    /// Zero-allocation vocal settings configuration
    #[inline]
    pub fn vocal_settings(
        mut self,
        config: impl Into<hashbrown::HashMap<&'static str, &'static str>>,
    ) -> Self {
        let config_map = config.into();
        for (k, v) in config_map {
            self.vocal_settings.insert(k.to_string(), v.to_string());
        }
        self
    }
}
