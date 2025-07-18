#![doc = r" Auto-generated voice definitions from ElevenLabs API"]
#![doc = r""]
#![doc = r" This file is automatically generated by build.rs and should not be edited manually."]
#![doc = r" Voice definitions are cached for 24 hours and refreshed automatically."]
use std::fmt;
#[doc = r" All available ElevenLabs voices with strong typing"]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Voice {
    #[doc = "A middle-aged female with an African-American accent. Calm with a hint of rasp."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Aria,
    #[doc = "Young adult woman with a confident and warm, mature quality and a reassuring, professional tone."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Sarah,
    #[doc = "This young adult female voice delivers sunny enthusiasm with a quirky attitude."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Laura,
    #[doc = "A young Australian male with a confident and energetic voice."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Charlie,
    #[doc = "Warm resonance that instantly captivates listeners."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    George,
    #[doc = "Deceptively gravelly, yet unsettling edge."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Callum,
    #[doc = "A relaxed, neutral voice ready for narrations or conversational projects."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    River,
    #[doc = "A young adult with energy and warmth - suitable for reels and shorts."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Liam,
    #[doc = "Sensual and raspy, she's ready to voice your temptress in video games."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Charlotte,
    #[doc = "Clear and engaging, friendly woman with a British accent suitable for e-learning."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Alice,
    #[doc = "A professional woman with a pleasing alto pitch. Suitable for many use cases."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Matilda,
    #[doc = "Conversational and laid back."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Will,
    #[doc = "Young and popular, this playful American female voice is perfect for trendy content."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Jessica,
    #[doc = "A smooth tenor pitch from a man in his 40s - perfect for agentic use cases."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Eric,
    #[doc = "Natural and real, this down-to-earth voice is great across many use-cases."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Chris,
    #[doc = "Middle-aged man with a resonant and comforting tone. Great for narrations and advertisements."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Brian,
    #[doc = "A strong voice perfect for delivering a professional broadcast or news story."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Daniel,
    #[doc = "Velvety British female voice delivers news and narrations with warmth and clarity."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Lily,
    #[doc = "Friendly and comforting voice ready to narrate your stories."]
    #[doc = r""]
    # [doc = concat ! ("Category: " , "premade")]
    Bill,
}
#[doc = r" Voice information including metadata and default settings"]
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
    #[doc = r" Get the ElevenLabs voice ID for this voice"]
    pub fn id(self) -> &'static str {
        match self {
            Voice::Aria => "9BWtsMINqrJLrRacOk9x",
            Voice::Sarah => "EXAVITQu4vr4xnSDxMaL",
            Voice::Laura => "FGY2WhTYpPnrIDTdsKH5",
            Voice::Charlie => "IKne3meq5aSn9XLyUdCD",
            Voice::George => "JBFqnCBsd6RMkjVDRZzb",
            Voice::Callum => "N2lVS1w4EtoT3dr4eOWO",
            Voice::River => "SAz9YHcvj6GT2YYXdXww",
            Voice::Liam => "TX3LPaxmHKxFdv7VOQHJ",
            Voice::Charlotte => "XB0fDUnXU5powFXDhCwa",
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
    #[doc = r" Get complete voice information"]
    pub fn info(self) -> VoiceInfo {
        match self {
            Voice::Aria => VoiceInfo {
                id: "9BWtsMINqrJLrRacOk9x",
                name: "Aria",
                category: "premade",
                description: "A middle-aged female with an African-American accent. Calm with a hint of rasp.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/9BWtsMINqrJLrRacOk9x/405766b8-1f4e-4d3c-aba1-6f25333823ec.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Sarah => VoiceInfo {
                id: "EXAVITQu4vr4xnSDxMaL",
                name: "Sarah",
                category: "premade",
                description: "Young adult woman with a confident and warm, mature quality and a reassuring, professional tone.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/EXAVITQu4vr4xnSDxMaL/01a3e33c-6e99-4ee7-8543-ff2216a32186.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Laura => VoiceInfo {
                id: "FGY2WhTYpPnrIDTdsKH5",
                name: "Laura",
                category: "premade",
                description: "This young adult female voice delivers sunny enthusiasm with a quirky attitude.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/FGY2WhTYpPnrIDTdsKH5/67341759-ad08-41a5-be6e-de12fe448618.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Charlie => VoiceInfo {
                id: "IKne3meq5aSn9XLyUdCD",
                name: "Charlie",
                category: "premade",
                description: "A young Australian male with a confident and energetic voice.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/IKne3meq5aSn9XLyUdCD/102de6f2-22ed-43e0-a1f1-111fa75c5481.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::George => VoiceInfo {
                id: "JBFqnCBsd6RMkjVDRZzb",
                name: "George",
                category: "premade",
                description: "Warm resonance that instantly captivates listeners.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/JBFqnCBsd6RMkjVDRZzb/e6206d1a-0721-4787-aafb-06a6e705cac5.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Callum => VoiceInfo {
                id: "N2lVS1w4EtoT3dr4eOWO",
                name: "Callum",
                category: "premade",
                description: "Deceptively gravelly, yet unsettling edge.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/N2lVS1w4EtoT3dr4eOWO/ac833bd8-ffda-4938-9ebc-b0f99ca25481.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::River => VoiceInfo {
                id: "SAz9YHcvj6GT2YYXdXww",
                name: "River",
                category: "premade",
                description: "A relaxed, neutral voice ready for narrations or conversational projects.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/SAz9YHcvj6GT2YYXdXww/e6c95f0b-2227-491a-b3d7-2249240decb7.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Liam => VoiceInfo {
                id: "TX3LPaxmHKxFdv7VOQHJ",
                name: "Liam",
                category: "premade",
                description: "A young adult with energy and warmth - suitable for reels and shorts.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/TX3LPaxmHKxFdv7VOQHJ/63148076-6363-42db-aea8-31424308b92c.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Charlotte => VoiceInfo {
                id: "XB0fDUnXU5powFXDhCwa",
                name: "Charlotte",
                category: "premade",
                description: "Sensual and raspy, she's ready to voice your temptress in video games.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/XB0fDUnXU5powFXDhCwa/942356dc-f10d-4d89-bda5-4f8505ee038b.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Alice => VoiceInfo {
                id: "Xb7hH8MSUJpSbSDYk0k2",
                name: "Alice",
                category: "premade",
                description: "Clear and engaging, friendly woman with a British accent suitable for e-learning.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/Xb7hH8MSUJpSbSDYk0k2/d10f7534-11f6-41fe-a012-2de1e482d336.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Matilda => VoiceInfo {
                id: "XrExE9yKIg1WjnnlVkGX",
                name: "Matilda",
                category: "premade",
                description: "A professional woman with a pleasing alto pitch. Suitable for many use cases.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/XrExE9yKIg1WjnnlVkGX/b930e18d-6b4d-466e-bab2-0ae97c6d8535.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Will => VoiceInfo {
                id: "bIHbv24MWmeRgasZH58o",
                name: "Will",
                category: "premade",
                description: "Conversational and laid back.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/bIHbv24MWmeRgasZH58o/8caf8f3d-ad29-4980-af41-53f20c72d7a4.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Jessica => VoiceInfo {
                id: "cgSgspJ2msm6clMCkdW9",
                name: "Jessica",
                category: "premade",
                description: "Young and popular, this playful American female voice is perfect for trendy content.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/cgSgspJ2msm6clMCkdW9/56a97bf8-b69b-448f-846c-c3a11683d45a.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Eric => VoiceInfo {
                id: "cjVigY5qzO86Huf0OWal",
                name: "Eric",
                category: "premade",
                description: "A smooth tenor pitch from a man in his 40s - perfect for agentic use cases.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/cjVigY5qzO86Huf0OWal/d098fda0-6456-4030-b3d8-63aa048c9070.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Chris => VoiceInfo {
                id: "iP95p4xoKVk53GoZ742B",
                name: "Chris",
                category: "premade",
                description: "Natural and real, this down-to-earth voice is great across many use-cases.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/iP95p4xoKVk53GoZ742B/3f4bde72-cc48-40dd-829f-57fbf906f4d7.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Brian => VoiceInfo {
                id: "nPczCjzI2devNBz1zQrb",
                name: "Brian",
                category: "premade",
                description: "Middle-aged man with a resonant and comforting tone. Great for narrations and advertisements.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/nPczCjzI2devNBz1zQrb/2dd3e72c-4fd3-42f1-93ea-abc5d4e5aa1d.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Daniel => VoiceInfo {
                id: "onwK4e9ZLuTAKqWW03F9",
                name: "Daniel",
                category: "premade",
                description: "A strong voice perfect for delivering a professional broadcast or news story.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/onwK4e9ZLuTAKqWW03F9/7eee0236-1a72-4b86-b303-5dcadc007ba9.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Lily => VoiceInfo {
                id: "pFZP5JQG7iQjIQuC4Bku",
                name: "Lily",
                category: "premade",
                description: "Velvety British female voice delivers news and narrations with warmth and clarity.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/pFZP5JQG7iQjIQuC4Bku/89b68b35-b3dd-4348-a84a-a3c13a3c2b30.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
            Voice::Bill => VoiceInfo {
                id: "pqHfZKP75CvOlQylNhV4",
                name: "Bill",
                category: "premade",
                description: "Friendly and comforting voice ready to narrate your stories.",
                preview_url: "https://storage.googleapis.com/eleven-public-prod/premade/voices/pqHfZKP75CvOlQylNhV4/d782b3ff-84ba-4029-848c-acf01285524d.mp3",
                default_stability: 0.75f64,
                default_similarity_boost: 0.75f64,
            },
        }
    }
    #[doc = r" Parse a voice from a string name (case-insensitive)"]
    pub fn from_name(name: &str) -> Option<Voice> {
        let name_lower = name.to_lowercase();
        match name_lower.as_str() {
            "aria" => Some(Voice::Aria),
            "sarah" => Some(Voice::Sarah),
            "laura" => Some(Voice::Laura),
            "charlie" => Some(Voice::Charlie),
            "george" => Some(Voice::George),
            "callum" => Some(Voice::Callum),
            "river" => Some(Voice::River),
            "liam" => Some(Voice::Liam),
            "charlotte" => Some(Voice::Charlotte),
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
    #[doc = r" Get all available voices"]
    pub fn all() -> Vec<Voice> {
        vec![
            Voice::Aria,
            Voice::Sarah,
            Voice::Laura,
            Voice::Charlie,
            Voice::George,
            Voice::Callum,
            Voice::River,
            Voice::Liam,
            Voice::Charlotte,
            Voice::Alice,
            Voice::Matilda,
            Voice::Will,
            Voice::Jessica,
            Voice::Eric,
            Voice::Chris,
            Voice::Brian,
            Voice::Daniel,
            Voice::Lily,
            Voice::Bill,
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
#[doc = r" Default voices for backward compatibility"]
pub mod defaults {
    use super::Voice;
    pub const ARIA: Voice = Voice::Aria;
    pub const SARAH: Voice = Voice::Sarah;
    pub const LAURA: Voice = Voice::Laura;
    pub const CHARLIE: Voice = Voice::Charlie;
    pub const GEORGE: Voice = Voice::George;
    pub const CALLUM: Voice = Voice::Callum;
    pub const RIVER: Voice = Voice::River;
    pub const LIAM: Voice = Voice::Liam;
    pub const CHARLOTTE: Voice = Voice::Charlotte;
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
