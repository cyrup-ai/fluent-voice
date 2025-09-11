//! Build script to dynamically generate voice enums from ElevenLabs API
//!
//! This script:
//! 1. Queries ElevenLabs v2 API for all available voices
//! 2. Generates strongly typed enums using syn/quote
//! 3. Implements 24-hour caching to avoid excessive API calls
//! 4. Writes generated code to src/voice.rs

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use quote::quote;
use serde::{Deserialize, Serialize};
use syn::Ident;

const CACHE_DURATION_HOURS: u64 = 24;
const CACHE_FILE: &str = "target/voice_cache.json";
const GENERATED_FILE: &str = "src/voice.rs";

#[derive(Debug, Serialize, Deserialize)]
struct VoiceCache {
    timestamp: u64,
    voices: Vec<ElevenLabsVoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ElevenLabsVoice {
    voice_id: String,
    name: String,
    category: Option<String>,
    description: Option<String>,
    labels: Option<HashMap<String, String>>,
    preview_url: Option<String>,
    available_for_tiers: Option<Vec<String>>,
    settings: Option<VoiceSettings>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VoiceSettings {
    stability: Option<f64>,
    similarity_boost: Option<f64>,
    style: Option<f64>,
    use_speaker_boost: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VoicesResponse {
    voices: Vec<ElevenLabsVoice>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");

    // Check if we need to refresh the cache
    let should_refresh = should_refresh_cache()?;

    let voices = if should_refresh {
        println!("Fetching voices from ElevenLabs API...");
        let voices = fetch_voices_from_api().await?;
        save_voice_cache(&voices)?;
        voices
    } else {
        println!("Using cached voices...");
        load_voice_cache()?.voices
    };

    println!("Generating voice enums for {} voices...", voices.len());
    generate_voice_code(&voices)?;

    println!("Voice code generation complete!");
    Ok(())
}

fn should_refresh_cache() -> Result<bool, Box<dyn std::error::Error>> {
    if !Path::new(CACHE_FILE).exists() {
        return Ok(true);
    }

    let cache_content = fs::read_to_string(CACHE_FILE)?;
    let cache: VoiceCache = serde_json::from_str(&cache_content)?;

    let current_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    let cache_age_hours = (current_time - cache.timestamp) / 3600;

    Ok(cache_age_hours >= CACHE_DURATION_HOURS)
}

async fn fetch_voices_from_api() -> Result<Vec<ElevenLabsVoice>, Box<dyn std::error::Error>> {
    // Try to fetch from API, but fall back to static voices if it fails
    match try_fetch_from_api().await {
        Ok(voices) => {
            println!(
                "Successfully fetched {} voices from ElevenLabs API",
                voices.len()
            );
            Ok(voices)
        }
        Err(e) => {
            println!("Failed to fetch from API ({e}), using fallback voices");
            Ok(get_fallback_voices())
        }
    }
}

async fn try_fetch_from_api() -> Result<Vec<ElevenLabsVoice>, Box<dyn std::error::Error>> {
    let api_key = env::var("ELEVENLABS_API_KEY")
        .or_else(|_| env::var("ELEVEN_API_KEY"))
        .map_err(|_| "ELEVENLABS_API_KEY or ELEVEN_API_KEY environment variable required for voice generation")?;

    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(false) // Keep security but handle cert errors gracefully
        .timeout(std::time::Duration::from_secs(10)) // Quick timeout
        .build()?;

    let response = client
        .get("https://api.elevenlabs.io/v1/voices")
        .header("xi-api-key", &api_key)
        .header("Content-Type", "application/json")
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(format!("API request failed: {}", response.status()).into());
    }

    let voices_response: VoicesResponse = response.json().await?;
    Ok(voices_response.voices)
}

fn get_fallback_voices() -> Vec<ElevenLabsVoice> {
    // Static list of popular ElevenLabs voices for offline builds
    vec![
        ElevenLabsVoice {
            voice_id: "21m00Tcm4TlvDq8ikWAM".to_string(),
            name: "Rachel".to_string(),
            category: Some("Generated".to_string()),
            description: Some("Young, cheerful American female voice".to_string()),
            labels: None,
            preview_url: Some("https://storage.googleapis.com/eleven-public-prod/premade/voices/21m00Tcm4TlvDq8ikWAM/48e4e68e-e940-4fd7-891f-91b20ce01c41.mp3".to_string()),
            available_for_tiers: Some(vec!["free".to_string(), "starter".to_string(), "creator".to_string(), "pro".to_string()]),
            settings: Some(VoiceSettings {
                stability: Some(0.75),
                similarity_boost: Some(0.75),
                style: Some(0.0),
                use_speaker_boost: Some(true),
            }),
        },
        ElevenLabsVoice {
            voice_id: "AZnzlk1XvdvUeBnXmlld".to_string(),
            name: "Domi".to_string(),
            category: Some("Generated".to_string()),
            description: Some("Strong, confident female voice".to_string()),
            labels: None,
            preview_url: Some("https://storage.googleapis.com/eleven-public-prod/premade/voices/AZnzlk1XvdvUeBnXmlld/d5b76ce8-8e5e-4f96-aa78-3b54b0fb4af4.mp3".to_string()),
            available_for_tiers: Some(vec!["free".to_string(), "starter".to_string(), "creator".to_string(), "pro".to_string()]),
            settings: Some(VoiceSettings {
                stability: Some(0.75),
                similarity_boost: Some(0.75),
                style: Some(0.0),
                use_speaker_boost: Some(true),
            }),
        },
        ElevenLabsVoice {
            voice_id: "EXAVITQu4vr4xnSDxMaL".to_string(),
            name: "Bella".to_string(),
            category: Some("Generated".to_string()),
            description: Some("Warm, friendly female voice".to_string()),
            labels: None,
            preview_url: Some("https://storage.googleapis.com/eleven-public-prod/premade/voices/EXAVITQu4vr4xnSDxMaL/e8c86b0a-e1c9-4b8a-9036-30b0f0d41dc5.mp3".to_string()),
            available_for_tiers: Some(vec!["free".to_string(), "starter".to_string(), "creator".to_string(), "pro".to_string()]),
            settings: Some(VoiceSettings {
                stability: Some(0.75),
                similarity_boost: Some(0.75),
                style: Some(0.0),
                use_speaker_boost: Some(true),
            }),
        },
        ElevenLabsVoice {
            voice_id: "ErXwobaYiN019PkySvjV".to_string(),
            name: "Antoni".to_string(),
            category: Some("Generated".to_string()),
            description: Some("Professional, articulate male voice".to_string()),
            labels: None,
            preview_url: Some("https://storage.googleapis.com/eleven-public-prod/premade/voices/ErXwobaYiN019PkySvjV/bb50e5bd-0844-44a0-9a72-7e7c5f71bee3.mp3".to_string()),
            available_for_tiers: Some(vec!["free".to_string(), "starter".to_string(), "creator".to_string(), "pro".to_string()]),
            settings: Some(VoiceSettings {
                stability: Some(0.75),
                similarity_boost: Some(0.75),
                style: Some(0.0),
                use_speaker_boost: Some(true),
            }),
        },
        ElevenLabsVoice {
            voice_id: "VR6AewLTigWG4xSOukaG".to_string(),
            name: "Arnold".to_string(),
            category: Some("Generated".to_string()),
            description: Some("Deep, authoritative male voice".to_string()),
            labels: None,
            preview_url: Some("https://storage.googleapis.com/eleven-public-prod/premade/voices/VR6AewLTigWG4xSOukaG/f752a3bb-86ce-46f6-ae5c-2b3b3c7ba4ca.mp3".to_string()),
            available_for_tiers: Some(vec!["free".to_string(), "starter".to_string(), "creator".to_string(), "pro".to_string()]),
            settings: Some(VoiceSettings {
                stability: Some(0.75),
                similarity_boost: Some(0.75),
                style: Some(0.0),
                use_speaker_boost: Some(true),
            }),
        },
    ]
}

fn save_voice_cache(voices: &[ElevenLabsVoice]) -> Result<(), Box<dyn std::error::Error>> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    let cache = VoiceCache {
        timestamp,
        voices: voices.to_vec(),
    };

    // Ensure target directory exists
    if let Some(parent) = Path::new(CACHE_FILE).parent() {
        fs::create_dir_all(parent)?;
    }

    let cache_json = serde_json::to_string_pretty(&cache)?;
    fs::write(CACHE_FILE, cache_json)?;

    Ok(())
}

fn load_voice_cache() -> Result<VoiceCache, Box<dyn std::error::Error>> {
    let cache_content = fs::read_to_string(CACHE_FILE)?;
    let cache: VoiceCache = serde_json::from_str(&cache_content)?;
    Ok(cache)
}

fn generate_voice_code(voices: &[ElevenLabsVoice]) -> Result<(), Box<dyn std::error::Error>> {
    let mut voice_variants = Vec::new();
    let mut voice_idents = Vec::new();
    let mut voice_id_matches = Vec::new();
    let mut voice_name_matches = Vec::new();
    let mut voice_info_matches = Vec::new();
    let mut default_constants = Vec::new();

    for voice in voices {
        // Create a valid Rust identifier from the voice name
        let variant_name = sanitize_identifier(&voice.name);
        let variant_ident = syn::parse_str::<Ident>(&variant_name)?;

        // Build the enum variant with documentation
        let description = voice.description.as_deref().unwrap_or("ElevenLabs voice");
        let category = voice.category.as_deref().unwrap_or("General");

        voice_variants.push(quote! {
            #[doc = #description]
            ///
            #[doc = concat!("Category: ", #category)]
            #variant_ident
        });

        // Keep track of just the identifiers for the all() method
        voice_idents.push(variant_ident.clone());

        // Add match arms for voice ID lookup
        let voice_id = &voice.voice_id;
        voice_id_matches.push(quote! {
            Voice::#variant_ident => #voice_id
        });

        // Add match arms for voice name lookup
        let voice_name = &voice.name;
        let voice_name_lower = voice.name.to_lowercase();
        voice_name_matches.push(quote! {
            #voice_name_lower => Some(Voice::#variant_ident)
        });

        // Add match arms for voice info
        let preview_url = voice.preview_url.as_deref().unwrap_or("");
        let stability = voice
            .settings
            .as_ref()
            .and_then(|s| s.stability)
            .unwrap_or(0.75);
        let similarity_boost = voice
            .settings
            .as_ref()
            .and_then(|s| s.similarity_boost)
            .unwrap_or(0.75);

        voice_info_matches.push(quote! {
            Voice::#variant_ident => VoiceInfo {
                id: #voice_id,
                name: #voice_name,
                category: #category,
                description: #description,
                preview_url: #preview_url,
                default_stability: #stability,
                default_similarity_boost: #similarity_boost,
            }
        });

        // Add to defaults module for backward compatibility
        let const_name = variant_name.to_uppercase();
        let const_ident = syn::parse_str::<Ident>(&const_name)?;
        default_constants.push(quote! {
            pub const #const_ident: Voice = Voice::#variant_ident;
        });
    }

    // Generate the complete voice module
    let generated_code = quote! {
        //! Auto-generated voice definitions from ElevenLabs API
        //!
        //! This file is automatically generated by build.rs and should not be edited manually.
        //! Voice definitions are cached for 24 hours and refreshed automatically.

        use std::fmt;

        /// All available ElevenLabs voices with strong typing
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Voice {
            #(#voice_variants,)*
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
                    #(#voice_id_matches,)*
                }
            }

            /// Get complete voice information
            pub fn info(self) -> VoiceInfo {
                match self {
                    #(#voice_info_matches,)*
                }
            }

            /// Parse a voice from a string name (case-insensitive)
            pub fn from_name(name: &str) -> Option<Voice> {
                let name_lower = name.to_lowercase();
                match name_lower.as_str() {
                    #(#voice_name_matches,)*
                    _ => None,
                }
            }

            /// Get all available voices
            pub fn all() -> Vec<Voice> {
                vec![
                    #(Voice::#voice_idents,)*
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

            #(#default_constants)*
        }
    };

    // Write the generated code to src/voice.rs
    let formatted_code = format!("{generated_code}");
    fs::write(GENERATED_FILE, formatted_code)?;

    Ok(())
}

fn sanitize_identifier(name: &str) -> String {
    // Convert voice name to a valid Rust identifier
    let mut result = String::new();
    let mut capitalize_next = true;

    for ch in name.chars() {
        if ch.is_alphanumeric() {
            if capitalize_next {
                result.push(ch.to_ascii_uppercase());
                capitalize_next = false;
            } else {
                result.push(ch);
            }
        } else if ch.is_whitespace() || ch == '-' || ch == '_' {
            capitalize_next = true;
        }
        // Skip other special characters
    }

    // Ensure it starts with a letter and isn't a Rust keyword
    if result.is_empty() || result.chars().next().unwrap().is_numeric() {
        result = format!("Voice{result}");
    }

    // Handle Rust keywords
    match result.as_str() {
        "Self" | "self" | "super" | "crate" | "Type" | "type" => {
            format!("{result}Voice")
        }
        _ => result,
    }
}
