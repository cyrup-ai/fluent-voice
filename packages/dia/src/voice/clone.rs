//! Voice clone implementation - the core voice representation

use crate::Tensor;
use anyhow::Result;
use std::sync::Arc;

use super::codec::VoiceData;
use super::{PitchNote, VocalSpeedMod, VoicePersona, VoiceTimber};

/// A cloned voice with all its characteristics
#[derive(Clone)]
pub struct VoiceClone {
    /// Unique identifier for this voice
    pub id: String,

    /// The encoded voice data
    pub voice_data: Arc<VoiceData>,

    /// Voice timber modification
    pub timber: Option<VoiceTimber>,

    /// Personality traits
    pub personas: Vec<VoicePersona>,

    /// Speed modifier
    pub speed: Option<VocalSpeedMod>,

    /// Pitch range (min, max)
    pub pitch_range: Option<(PitchNote, PitchNote)>,

    /// Segment configuration for voice prompting
    pub segment_config: SegmentConfig,
}

/// Configuration for how to extract voice segments for prompting
#[derive(Clone)]
pub struct SegmentConfig {
    /// Preferred duration for voice segments (in seconds)
    pub duration: f32,

    /// Where to start extracting from (0.0 = beginning, 1.0 = end)
    pub position: f32,

    /// Whether to use random segments
    pub randomize: bool,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            duration: 7.0, // 7 seconds is a good balance
            position: 0.2, // Skip first 20% (often has noise/silence)
            randomize: false,
        }
    }
}

impl VoiceClone {
    /// Create a new voice clone from voice data
    pub fn new(id: impl Into<String>, voice_data: Arc<VoiceData>) -> Self {
        Self {
            id: id.into(),
            voice_data,
            timber: None,
            personas: Vec::new(),
            speed: None,
            pitch_range: None,
            segment_config: SegmentConfig::default(),
        }
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
    pub fn with_pitch_range(mut self, min: PitchNote, max: PitchNote) -> Self {
        self.pitch_range = Some((min, max));
        self
    }

    /// Configure how segments are extracted
    pub fn with_segment_config(mut self, config: SegmentConfig) -> Self {
        self.segment_config = config;
        self
    }

    /// Get the optimal audio prompt for this voice
    pub fn get_audio_prompt(&self) -> Result<Tensor> {
        let total_duration = self.voice_data.duration()?;

        // Calculate start time based on position
        let start_time = if self.segment_config.randomize {
            // Random position that ensures we get a full segment
            let max_start = (total_duration - self.segment_config.duration).max(0.0);
            rand::random::<f32>() * max_start
        } else {
            // Use configured position
            let max_start = (total_duration - self.segment_config.duration).max(0.0);
            self.segment_config.position * max_start
        };

        // Extract the segment
        self.voice_data
            .extract_segment(start_time, self.segment_config.duration)
    }

    /// Generate speaker tags for this voice
    pub fn generate_tags(&self) -> String {
        let mut tags = Vec::new();

        // Add timber tags
        if let Some(timber) = &self.timber {
            tags.push(format!("timber:{timber:?}").to_lowercase());
        }

        // Add persona tags
        for persona in &self.personas {
            tags.push(format!("persona:{persona:?}").to_lowercase());
        }

        // Add speed tags
        if let Some(speed) = &self.speed {
            tags.push(format!("speed:{speed:?}").to_lowercase());
        }

        // Format as hidden tags
        if tags.is_empty() {
            String::new()
        } else {
            format!("[{}]", tags.join(","))
        }
    }
}

/// Builder for creating voice clones
pub struct VoiceCloneBuilder {
    id: String,
    voice_data: Option<Arc<VoiceData>>,
    timber: Option<VoiceTimber>,
    personas: Vec<VoicePersona>,
    speed: Option<VocalSpeedMod>,
    pitch_range: Option<(PitchNote, PitchNote)>,
    segment_config: SegmentConfig,
}

impl VoiceCloneBuilder {
    /// Create a new builder with the given ID
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            voice_data: None,
            timber: None,
            personas: Vec::new(),
            speed: None,
            pitch_range: None,
            segment_config: SegmentConfig::default(),
        }
    }

    /// Set the voice data
    pub fn voice_data(mut self, data: Arc<VoiceData>) -> Self {
        self.voice_data = Some(data);
        self
    }

    /// Set the timber
    pub fn timber(mut self, timber: VoiceTimber) -> Self {
        self.timber = Some(timber);
        self
    }

    /// Add a persona trait
    pub fn persona(mut self, persona: VoicePersona) -> Self {
        self.personas.push(persona);
        self
    }

    /// Set the speed
    pub fn speed(mut self, speed: VocalSpeedMod) -> Self {
        self.speed = Some(speed);
        self
    }

    /// Set the pitch range
    pub fn pitch_range(mut self, min: PitchNote, max: PitchNote) -> Self {
        self.pitch_range = Some((min, max));
        self
    }

    /// Set the segment configuration
    pub fn segment_config(mut self, config: SegmentConfig) -> Self {
        self.segment_config = config;
        self
    }

    /// Build the voice clone
    pub fn build(self) -> Result<VoiceClone> {
        let voice_data = self
            .voice_data
            .ok_or_else(|| anyhow::anyhow!("Voice data is required"))?;

        Ok(VoiceClone {
            id: self.id,
            voice_data,
            timber: self.timber,
            personas: self.personas,
            speed: self.speed,
            pitch_range: self.pitch_range,
            segment_config: self.segment_config,
        })
    }
}
