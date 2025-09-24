//! Auto-generated voice definitions from ElevenLabs API
//!
//! This file contains all 22 available ElevenLabs voices with strong typing.

use std::fmt;

/// All available ElevenLabs voices with strong typing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Voice {
    /// Young, cheerful American female voice
    Rachel,
    /// Deep, authoritative male voice  
    Clyde,
    /// Confident, articulate male voice
    Roger,
    /// Warm, friendly female voice
    Sarah,
    /// Professional female voice
    Laura,
    /// Strong, confident male voice
    Thomas,
    /// Energetic, youthful male voice
    Charlie,
    /// Mature, distinguished male voice
    George,
    /// British, articulate male voice
    Callum,
    /// Calm, soothing unisex voice
    River,
    /// Young, enthusiastic male voice
    Harry,
    /// Friendly, approachable male voice
    Liam,
    /// Clear, professional female voice
    Alice,
    /// Young, bright female voice
    Matilda,
    /// Casual, conversational male voice
    Will,
    /// Sophisticated female voice
    Jessica,
    /// Authoritative male voice
    Eric,
    /// Friendly, reliable male voice
    Chris,
    /// Professional, trustworthy male voice
    Brian,
    /// Warm, engaging male voice
    Daniel,
    /// Sweet, gentle female voice
    Lily,
    /// Experienced, wise male voice
    Bill,
}

/// Voice information including metadata and default settings
#[derive(Debug, Clone)]
pub struct VoiceInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub category: &'static str,
    pub description: &'static str,
    pub preview_url: &'static str,
    pub default_stability: f64,
    pub default_similarity_boost: f64,
}

impl Voice {
    /// Get the ElevenLabs voice ID for this voice
    pub fn id(self) -> &'static str {
        match self {
            Voice::Rachel => "21m00Tcm4TlvDq8ikWAM",
            Voice::Clyde => "2EiwWnXFnvU5JabPnv8n",
            Voice::Roger => "CYw3kZ02Hs0563khs1Fj",
            Voice::Sarah => "EXAVITQu4vr4xnSDxMaL",
            Voice::Laura => "FGY2WhTYpPnrIDTdsKH5",
            Voice::Thomas => "GBv7mTt0atIp3Br8iCZE",
            Voice::Charlie => "IKne3meq5aSn9XLyUdCD",
            Voice::George => "JBFqnCBsd6RMkjVDRZzb",
            Voice::Callum => "N2lVS1w4EtoT3dr4eOWO",
            Voice::River => "SAz9YHcvj6GT2YYXdXww",
            Voice::Harry => "SOYHLrjzK2X1ezoPC6cr",
            Voice::Liam => "TX3LPaxmHKxFdv7VOQHJ",
            Voice::Alice => "Xb7hH8MSUJpSbSDYk0k2",
            Voice::Matilda => "XrExE9yKIg1WjnnlVkGX",
            Voice::Will => "bIHbv24MWmeRgasZH58o",
            Voice::Jessica => "cgSgspJ2msm6clMCkdW9",
            Voice::Eric => "cjVigY5qzO86Huf0OWal",
            Voice::Chris => "iP95p4xoKVk53GoZ742B",
            Voice::Brian => "nPczCjzI2devNBz1zQrb",
            Voice::Daniel => "onwK4e9ZLuTAKqWW03F9",
            Voice::Lily => "pFZP5JQG7iQjIQuC4Bku",
            Voice::Bill => "pqHfZKP75CvOlQylNhV4",
        }
    }

    /// Get complete voice information
    pub fn info(self) -> VoiceInfo {
        match self {
            Voice::Rachel => VoiceInfo {
                id: "21m00Tcm4TlvDq8ikWAM",
                name: "Rachel",
                category: "Generated",
                description: "Young, cheerful American female voice",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/21m00Tcm4TlvDq8ikWAM/48e4e68e-e940-4fd7-891f-91b20ce01c41.mp3",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Clyde => VoiceInfo {
                id: "2EiwWnXFnvU5JabPnv8n",
                name: "Clyde",
                category: "Generated",
                description: "Deep, authoritative male voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Roger => VoiceInfo {
                id: "CYw3kZ02Hs0563khs1Fj",
                name: "Roger",
                category: "Generated", 
                description: "Confident, articulate male voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Sarah => VoiceInfo {
                id: "EXAVITQu4vr4xnSDxMaL",
                name: "Sarah",
                category: "Generated",
                description: "Warm, friendly female voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Laura => VoiceInfo {
                id: "FGY2WhTYpPnrIDTdsKH5",
                name: "Laura",
                category: "Generated",
                description: "Professional female voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Thomas => VoiceInfo {
                id: "GBv7mTt0atIp3Br8iCZE",
                name: "Thomas",
                category: "Generated",
                description: "Strong, confident male voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Charlie => VoiceInfo {
                id: "IKne3meq5aSn9XLyUdCD",
                name: "Charlie",
                category: "Generated",
                description: "Energetic, youthful male voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::George => VoiceInfo {
                id: "JBFqnCBsd6RMkjVDRZzb",
                name: "George",
                category: "Generated",
                description: "Mature, distinguished male voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Callum => VoiceInfo {
                id: "N2lVS1w4EtoT3dr4eOWO",
                name: "Callum",
                category: "Generated",
                description: "British, articulate male voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::River => VoiceInfo {
                id: "SAz9YHcvj6GT2YYXdXww",
                name: "River",
                category: "Generated",
                description: "Calm, soothing unisex voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Harry => VoiceInfo {
                id: "SOYHLrjzK2X1ezoPC6cr",
                name: "Harry",
                category: "Generated",
                description: "Young, enthusiastic male voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Liam => VoiceInfo {
                id: "TX3LPaxmHKxFdv7VOQHJ",
                name: "Liam",
                category: "Generated",
                description: "Friendly, approachable male voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Alice => VoiceInfo {
                id: "Xb7hH8MSUJpSbSDYk0k2",
                name: "Alice",
                category: "Generated",
                description: "Clear, professional female voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Matilda => VoiceInfo {
                id: "XrExE9yKIg1WjnnlVkGX",
                name: "Matilda",
                category: "Generated",
                description: "Young, bright female voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Will => VoiceInfo {
                id: "bIHbv24MWmeRgasZH58o",
                name: "Will",
                category: "Generated",
                description: "Casual, conversational male voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Jessica => VoiceInfo {
                id: "cgSgspJ2msm6clMCkdW9",
                name: "Jessica",
                category: "Generated",
                description: "Sophisticated female voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Eric => VoiceInfo {
                id: "cjVigY5qzO86Huf0OWal",
                name: "Eric",
                category: "Generated",
                description: "Authoritative male voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Chris => VoiceInfo {
                id: "iP95p4xoKVk53GoZ742B",
                name: "Chris",
                category: "Generated",
                description: "Friendly, reliable male voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Brian => VoiceInfo {
                id: "nPczCjzI2devNBz1zQrb",
                name: "Brian",
                category: "Generated",
                description: "Professional, trustworthy male voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Daniel => VoiceInfo {
                id: "onwK4e9ZLuTAKqWW03F9",
                name: "Daniel",
                category: "Generated",
                description: "Warm, engaging male voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Lily => VoiceInfo {
                id: "pFZP5JQG7iQjIQuC4Bku",
                name: "Lily",
                category: "Generated",
                description: "Sweet, gentle female voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
            Voice::Bill => VoiceInfo {
                id: "pqHfZKP75CvOlQylNhV4",
                name: "Bill",
                category: "Generated",
                description: "Experienced, wise male voice",
                preview_url: "",
                default_stability: 0.75,
                default_similarity_boost: 0.75,
            },
        }
    }

    /// Parse a voice from a string name (case-insensitive)
    pub fn from_name(name: &str) -> Option<Voice> {
        let name_lower = name.to_lowercase();
        match name_lower.as_str() {
            "rachel" => Some(Voice::Rachel),
            "clyde" => Some(Voice::Clyde),
            "roger" => Some(Voice::Roger),
            "sarah" => Some(Voice::Sarah),
            "laura" => Some(Voice::Laura),
            "thomas" => Some(Voice::Thomas),
            "charlie" => Some(Voice::Charlie),
            "george" => Some(Voice::George),
            "callum" => Some(Voice::Callum),
            "river" => Some(Voice::River),
            "harry" => Some(Voice::Harry),
            "liam" => Some(Voice::Liam),
            "alice" => Some(Voice::Alice),
            "matilda" => Some(Voice::Matilda),
            "will" => Some(Voice::Will),
            "jessica" => Some(Voice::Jessica),
            "eric" => Some(Voice::Eric),
            "chris" => Some(Voice::Chris),
            "brian" => Some(Voice::Brian),
            "daniel" => Some(Voice::Daniel),
            "lily" => Some(Voice::Lily),
            "bill" => Some(Voice::Bill),
            _ => None,
        }
    }

    /// Get all available voices
    pub fn all() -> Vec<Voice> {
        vec![
            Voice::Rachel, Voice::Clyde, Voice::Roger, Voice::Sarah,
            Voice::Laura, Voice::Thomas, Voice::Charlie, Voice::George,
            Voice::Callum, Voice::River, Voice::Harry, Voice::Liam,
            Voice::Alice, Voice::Matilda, Voice::Will, Voice::Jessica,
            Voice::Eric, Voice::Chris, Voice::Brian, Voice::Daniel,
            Voice::Lily, Voice::Bill,
        ]
    }
}

impl fmt::Display for Voice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.info().name)
    }
}

impl From<Voice> for String {
    fn from(voice: Voice) -> String {
        voice.id().to_string()
    }
}

/// Default voices for backward compatibility
pub mod defaults {
    use super::Voice;

    pub const RACHEL: Voice = Voice::Rachel;
    pub const CLYDE: Voice = Voice::Clyde;
    pub const ROGER: Voice = Voice::Roger;
    pub const SARAH: Voice = Voice::Sarah;
    pub const LAURA: Voice = Voice::Laura;
    pub const THOMAS: Voice = Voice::Thomas;
    pub const CHARLIE: Voice = Voice::Charlie;
    pub const GEORGE: Voice = Voice::George;
    pub const CALLUM: Voice = Voice::Callum;
    pub const RIVER: Voice = Voice::River;
    pub const HARRY: Voice = Voice::Harry;
    pub const LIAM: Voice = Voice::Liam;
    pub const ALICE: Voice = Voice::Alice;
    pub const MATILDA: Voice = Voice::Matilda;
    pub const WILL: Voice = Voice::Will;
    pub const JESSICA: Voice = Voice::Jessica;
    pub const ERIC: Voice = Voice::Eric;
    pub const CHRIS: Voice = Voice::Chris;
    pub const BRIAN: Voice = Voice::Brian;
    pub const DANIEL: Voice = Voice::Daniel;
    pub const LILY: Voice = Voice::Lily;
    pub const BILL: Voice = Voice::Bill;
}