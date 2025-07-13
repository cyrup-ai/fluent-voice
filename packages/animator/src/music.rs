use std::{num::ParseIntError, str::FromStr};

/// Musical tone representation (C, C#/Db, D, etc.)
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Tone {
    C,
    Db,
    D,
    Eb,
    E,
    F,
    Gb,
    G,
    Ab,
    A,
    Bb,
    B,
}

/// Error type for invalid tone parsing
#[derive(Debug, thiserror::Error)]
#[error("Invalid tone: {0}")]
pub struct ToneError(String);

/// A musical note consisting of a tone and octave
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct Note {
    tone: Tone,
    octave: u32,
}

/// Error types for note parsing
#[derive(Debug, thiserror::Error)]
pub enum NoteError {
    #[error("Invalid octave: {0}")]
    InvalidOctave(#[from] ParseIntError),
    #[error("Invalid note: {0}")]
    InvalidNote(#[from] ToneError),
}

impl FromStr for Note {
    type Err = NoteError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();

        // Find the split point between tone and octave
        let split_pos = trimmed
            .char_indices()
            .find(|(_, c)| c.is_ascii_digit())
            .map(|(i, _)| i)
            .unwrap_or(trimmed.len());

        let (tone_str, octave_str) = trimmed.split_at(split_pos);

        let tone = tone_str.parse::<Tone>()?;
        let octave = if octave_str.is_empty() {
            4 // Default to octave 4 (middle octave)
        } else {
            octave_str.parse::<u32>()?
        };

        Ok(Note { tone, octave })
    }
}

impl FromStr for Tone {
    type Err = ToneError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "C" => Ok(Tone::C),
            "C#" | "Db" => Ok(Tone::Db),
            "D" => Ok(Tone::D),
            "D#" | "Eb" => Ok(Tone::Eb),
            "E" => Ok(Tone::E),
            "F" => Ok(Tone::F),
            "F#" | "Gb" => Ok(Tone::Gb),
            "G" => Ok(Tone::G),
            "G#" | "Ab" => Ok(Tone::Ab),
            "A" => Ok(Tone::A),
            "A#" | "Bb" => Ok(Tone::Bb),
            "B" => Ok(Tone::B),
            _ => Err(ToneError(s.to_string())),
        }
    }
}

impl Note {
    /// Calculate the buffer size needed for one period of this note's waveform
    pub fn tune_buffer_size(&self, sample_rate: u32) -> u32 {
        let period = 1.0 / self.frequency();
        let buffer_size = (sample_rate as f32) * period;
        buffer_size.round() as u32
    }

    /// Get the frequency of this note in Hz
    pub fn frequency(&self) -> f32 {
        self.tone.freq(self.octave)
    }
}

impl Tone {
    /// Get the frequency of this tone at the given octave
    pub fn freq(&self, octave: u32) -> f32 {
        // Use A4 = 440 Hz as reference
        const A4_FREQ: f32 = 440.0;
        const A4_OCTAVE: i32 = 4;

        // Calculate semitone distance from A4
        let semitones_from_a = match self {
            Tone::C => -9,
            Tone::Db => -8,
            Tone::D => -7,
            Tone::Eb => -6,
            Tone::E => -5,
            Tone::F => -4,
            Tone::Gb => -3,
            Tone::G => -2,
            Tone::Ab => -1,
            Tone::A => 0,
            Tone::Bb => 1,
            Tone::B => 2,
        };

        // Calculate total semitone distance including octave difference
        let octave_diff = octave as i32 - A4_OCTAVE;
        let total_semitones = semitones_from_a + (octave_diff * 12);

        // Calculate frequency: f = f0 * 2^(n/12)
        A4_FREQ * 2.0_f32.powf(total_semitones as f32 / 12.0)
    }

    /// Get the semitone index (0-11) for this tone
    pub fn semitone_index(&self) -> u8 {
        match self {
            Tone::C => 0,
            Tone::Db => 1,
            Tone::D => 2,
            Tone::Eb => 3,
            Tone::E => 4,
            Tone::F => 5,
            Tone::Gb => 6,
            Tone::G => 7,
            Tone::Ab => 8,
            Tone::A => 9,
            Tone::Bb => 10,
            Tone::B => 11,
        }
    }
}
