//! Musical pitch notes for voice range

/// Musical note without octave
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Note {
    C,
    CSharp,
    D,
    DSharp,
    E,
    F,
    FSharp,
    G,
    GSharp,
    A,
    ASharp,
    B,
}

/// Octave number
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Octave {
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
}

/// Complete pitch note combining note and octave
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PitchNote {
    pub note: Note,
    pub octave: Octave,
}

impl PitchNote {
    pub const fn new(note: Note, octave: Octave) -> Self {
        Self { note, octave }
    }

    // Common notes as constants for convenience
    pub const A_FLAT_2: Self = Self::new(Note::GSharp, Octave::Two); // Ab2 = G#2
    pub const C3: Self = Self::new(Note::C, Octave::Three);
}
