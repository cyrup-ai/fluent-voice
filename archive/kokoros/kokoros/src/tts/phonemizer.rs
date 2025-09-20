use crate::tts::normalize;
use crate::tts::vocab::VOCAB;
use lazy_static::lazy_static;
use regex::Regex;
use thiserror::Error;

lazy_static! {
    static ref PHONEME_PATTERNS: Regex = Regex::new(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)").unwrap();
    static ref Z_PATTERN: Regex = Regex::new(r#" z(?=[;:,.!?¡¿—…"«»"" ]|$)"#).unwrap();
    static ref NINETY_PATTERN: Regex = Regex::new(r"(?<=nˈaɪn)ti(?!ː)").unwrap();
}

// Placeholder for the EspeakBackend struct
struct EspeakBackend {
    _language: String,
    _preserve_punctuation: bool,
    _with_stress: bool,
}

impl EspeakBackend {
    fn new(language: &str, preserve_punctuation: bool, with_stress: bool) -> Self {
        EspeakBackend {
            _language: language.to_string(),
            _preserve_punctuation: preserve_punctuation,
            _with_stress: with_stress,
        }
    }

    fn phonemize(&self, text: &[String]) -> Option<Vec<String>> {
        // Basic English phonemization implementation
        if text.is_empty() {
            return None;
        }

        let mut result = Vec::new();
        for input_text in text {
            let phonemes = self.text_to_phonemes(input_text);
            result.push(phonemes);
        }
        
        Some(result)
    }

    /// Convert English text to IPA phonemes
    fn text_to_phonemes(&self, text: &str) -> String {
        let text = text.to_lowercase();
        let mut phonemes = String::new();
        
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            if i > 0 {
                phonemes.push(' ');
            }
            
            // Basic phoneme mapping for common English patterns
            let word_phonemes = match *word {
                "the" => "ðə",
                "and" => "ænd",
                "to" => "tu",
                "a" => "ə",
                "in" => "ɪn",
                "is" => "ɪz",
                "it" => "ɪt",
                "you" => "ju",
                "that" => "ðæt",
                "he" => "hi",
                "was" => "wʌz",
                "for" => "fɔr",
                "on" => "ɔn",
                "are" => "ɑr",
                "as" => "æz",
                "with" => "wɪθ",
                "his" => "hɪz",
                "they" => "ðeɪ",
                "at" => "æt",
                "be" => "bi",
                "this" => "ðɪs",
                "have" => "hæv",
                "from" => "frʌm",
                "or" => "ɔr",
                "one" => "wʌn",
                "had" => "hæd",
                "by" => "baɪ",
                "word" => "wɜrd",
                "but" => "bʌt",
                "not" => "nɑt",
                "what" => "wʌt",
                "all" => "ɔl",
                "were" => "wɜr",
                "we" => "wi",
                "when" => "wɛn",
                "your" => "jʊr",
                "can" => "kæn",
                "said" => "sɛd",
                "there" => "ðɛr",
                "each" => "itʃ",
                "which" => "wɪtʃ",
                "do" => "du",
                "how" => "haʊ",
                "their" => "ðɛr",
                "if" => "ɪf",
                "will" => "wɪl",
                "up" => "ʌp",
                "other" => "ʌðər",
                "about" => "əbaʊt",
                "out" => "aʊt",
                "many" => "mɛni",
                "then" => "ðɛn",
                "them" => "ðɛm",
                "these" => "ðiz",
                "so" => "soʊ",
                "some" => "sʌm",
                "her" => "hɜr",
                "would" => "wʊd",
                "make" => "meɪk",
                "like" => "laɪk",
                "into" => "ɪntu",
                "him" => "hɪm",
                "time" => "taɪm",
                "has" => "hæz",
                "two" => "tu",
                "more" => "mɔr",
                "go" => "goʊ",
                "no" => "noʊ",
                "way" => "weɪ",
                "could" => "kʊd",
                "my" => "maɪ",
                "than" => "ðæn",
                "first" => "fɜrst",
                "water" => "wɔtər",
                "been" => "bɪn",
                "call" => "kɔl",
                "who" => "hu",
                "its" => "ɪts",
                "now" => "naʊ",
                "find" => "faɪnd",
                "long" => "lɔŋ",
                "down" => "daʊn",
                "day" => "deɪ",
                "did" => "dɪd",
                "get" => "gɛt",
                "come" => "kʌm",
                "made" => "meɪd",
                "may" => "meɪ",
                "part" => "pɑrt",
                _ => {
                    // For unknown words, apply basic letter-to-phoneme rules
                    self.apply_basic_phoneme_rules(word)
                }
            };
            phonemes.push_str(word_phonemes);
        }
        
        phonemes
    }

    /// Apply basic letter-to-phoneme conversion rules
    fn apply_basic_phoneme_rules(&self, word: &str) -> &str {
        // This is a simplified phoneme conversion
        // In a real implementation, this would use comprehensive phoneme rules
        match word.len() {
            0 => "",
            1..=3 => match word {
                word if word.ends_with('e') && word.len() > 1 => {
                    // Silent e rule - simplified
                    match &word[..word.len()-1] {
                        "tim" => "taɪm",
                        "mak" => "meɪk",
                        "tak" => "teɪk",
                        "giv" => "gɪv",
                        _ => word, // Fallback to original
                    }
                },
                _ => word, // Fallback to original for short words
            },
            _ => word, // Fallback to original for longer words
        }
    }
}

#[derive(Debug, Error)]
pub enum PhonemizerError {
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),
}

pub struct Phonemizer {
    lang: String,
    backend: EspeakBackend,
}

impl Phonemizer {
    pub fn new(lang: &str) -> Result<Self, PhonemizerError> {
        let backend = match lang {
            "a" => EspeakBackend::new("en-us", true, true),
            "b" => EspeakBackend::new("en-gb", true, true),
            unsupported => return Err(PhonemizerError::UnsupportedLanguage(unsupported.to_string())),
        };
        
        Ok(Phonemizer {
            lang: lang.to_string(),
            backend,
        })
    }

    pub fn phonemize(&self, text: &str, normalize: bool) -> String {
        let text = if normalize {
            normalize::normalize_text(text)
        } else {
            text.to_string()
        };

        // Assume phonemize returns Option<String>
        let mut ps = match self.backend.phonemize(&[text]) {
            Some(phonemes) => phonemes[0].clone(),
            None => String::new(),
        };

        // Apply kokoro-specific replacements
        ps = ps
            .replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ")
            .replace("kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ");

        // Apply character replacements
        ps = ps
            .replace("ʲ", "j")
            .replace("r", "ɹ")
            .replace("x", "k")
            .replace("ɬ", "l");

        // Apply regex patterns
        ps = PHONEME_PATTERNS.replace_all(&ps, " ").to_string();
        ps = Z_PATTERN.replace_all(&ps, "z").to_string();

        if self.lang == "a" {
            ps = NINETY_PATTERN.replace_all(&ps, "di").to_string();
        }

        // Filter characters present in vocabulary
        ps = ps.chars().filter(|&c| VOCAB.contains_key(&c)).collect();

        ps.trim().to_string()
    }
}
