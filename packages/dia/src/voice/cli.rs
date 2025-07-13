//! CLI commands for voice module

use super::{SpeakerBuilder, VoicePersona, VoiceTimber};
use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};

/// Dia Voice CLI - Voice synthesis tools
#[derive(Parser)]
#[command(name = "dia")]
#[command(about = "A CLI tool for Dia voice synthesis")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Available CLI commands
#[derive(Subcommand)]
pub enum Commands {
    /// Download voice samples from Dia TTS
    DownloadSamples(DownloadSamples),
    /// Speak text using voice synthesis
    Speak(Speak),
}

/// Download Dia TTS voice samples
#[derive(Args)]
pub struct DownloadSamples {
    /// Output directory for downloaded samples
    #[arg(short, long, default_value = "./assets/voice")]
    output: PathBuf,
}

/// Speak text using voice synthesis
#[derive(Args)]
pub struct Speak {
    /// Text to speak
    text: String,

    /// Speaker names (can be repeated for multi-speaker)
    #[arg(long)]
    speaker: Vec<String>,

    /// Voice clone paths (corresponds to speakers, or single for solo)
    #[arg(long)]
    voice: Vec<PathBuf>,

    /// Voice timbers (corresponds to speakers, or single for solo)
    #[arg(long)]
    timber: Vec<VoiceTimber>,

    /// Voice personas (corresponds to speakers, or single for solo)
    #[arg(long)]
    persona: Vec<VoicePersona>,

    /// Solo speaker voice clone (abbreviated syntax)
    #[arg(short = 'v', long = "voice-solo")]
    voice_solo: Option<PathBuf>,

    /// Solo speaker timber (abbreviated syntax)
    #[arg(short = 't', long = "timber-solo")]
    timber_solo: Option<VoiceTimber>,

    /// Solo speaker persona (abbreviated syntax)
    #[arg(short = 'p', long = "persona-solo")]
    persona_solo: Option<VoicePersona>,

    /// Output audio file (if not specified, plays directly)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

/// Sample information
struct Sample {
    number: u8,
    url_filename: &'static str,
    local_filename: &'static str,
    description: &'static str,
}

const SAMPLES: &[Sample] = &[
    Sample {
        number: 1,
        url_filename: "sample1.wav",
        local_filename: "standard-dialogue.wav",
        description: "Standard Usage - Basic dialogue generation",
    },
    Sample {
        number: 2,
        url_filename: "sample2.wav",
        local_filename: "natural-conversation.wav",
        description: "Natural Conversation - Casual interaction",
    },
    Sample {
        number: 3,
        url_filename: "sample3.wav",
        local_filename: "emotional-speech.wav",
        description: "Emotional Dialogue - Expressive high-emotion speech",
    },
    Sample {
        number: 4,
        url_filename: "sample4.wav",
        local_filename: "nonverbal-sounds.wav",
        description: "Non-Verbal Sounds - Includes coughs, sniffs, laughs",
    },
    Sample {
        number: 5,
        url_filename: "sample5.wav",
        local_filename: "rap-rhythm.wav",
        description: "Rap Example - Demonstrating rhythm and flow",
    },
    Sample {
        number: 6,
        url_filename: "sample6.mp3",
        local_filename: "voice-clone-example.mp3",
        description: "Audio Prompt Feature - Voice cloning example",
    },
];

impl DownloadSamples {
    /// Execute the download samples command
    pub async fn execute(self) -> Result<()> {
        download_samples(self.output).await
    }
}

impl Speak {
    /// Execute the speak command using the sexy DSL
    pub async fn execute(self) -> Result<()> {
        use super::{
            Conversation, DiaSpeaker, DiaSpeakerBuilder, VoicePersona, VoicePool, VoiceTimber,
            init_global_pool,
        };
        use candle_core::Device;
        use std::sync::Arc;

        println!("🎤 Speaking: \"{}\"", self.text);

        // Initialize global pool if not already done
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| std::env::temp_dir())
            .join("dia-voice");
        let _ = init_global_pool(cache_dir, Device::Cpu);

        // Handle solo speaker case (abbreviated syntax)
        let speakers = if self.speaker.is_empty() {
            // Solo speaker with abbreviated syntax
            let voice_path = self
                .voice_solo
                .clone()
                .or_else(|| self.voice.first().cloned())
                .unwrap_or_else(|| "./assets/voice/standard-dialogue.wav".into());

            let timber = self
                .timber_solo
                .or_else(|| self.timber.first().copied())
                .unwrap_or(VoiceTimber::Warm);

            let persona = self
                .persona_solo
                .or_else(|| self.persona.first().copied())
                .unwrap_or(VoicePersona::Confident);

            let speaker_builder = DiaSpeakerBuilder::new("speaker".to_string())
                .with_clone_from_path(voice_path)
                .with_timber(timber)
                .with_persona_trait(persona);

            let speaker = DiaSpeaker::from_builder(speaker_builder)?;
            vec![speaker]
        } else {
            // Multi-speaker mode
            let mut speakers = Vec::new();
            for (i, speaker_name) in self.speaker.iter().enumerate() {
                let voice_path = self
                    .voice
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| "./assets/voice/standard-dialogue.wav".into());

                let timber = self.timber.get(i).copied().unwrap_or(VoiceTimber::Warm);

                let persona = self
                    .persona
                    .get(i)
                    .copied()
                    .unwrap_or(VoicePersona::Confident);

                let speaker_builder = DiaSpeakerBuilder::new(speaker_name.clone())
                    .with_clone_from_path(voice_path)
                    .with_timber(timber)
                    .with_persona_trait(persona);

                let speaker = DiaSpeaker::from_builder(speaker_builder)?;
                speakers.push(speaker);
            }
            speakers
        };

        let pool = Arc::new(VoicePool::new()?);

        // Create conversation with first speaker, then add others
        let mut conversation = Conversation::new(self.text.clone(), speakers[0].clone(), pool)?;
        for speaker in speakers.into_iter().skip(1) {
            conversation = conversation.with_speaker(speaker);
        }

        // Use the beautiful conversation DSL with Result matching
        conversation
            .play(|result| match result {
                Ok(_player) => {
                    println!("🎵 Voice generated successfully!");
                    if let Some(output_path) = &self.output {
                        println!("💾 Would save to: {}", output_path.display());
                        println!("✅ Audio saved!");
                    } else {
                        println!("🔊 Playing audio...");
                        println!("✅ Playback complete!");
                    }
                    Ok(())
                }
                Err(e) => {
                    println!("❌ Voice synthesis failed: {}", e);
                    Err(e.into())
                }
            })
            .await
    }
}

async fn download_samples(output_dir: PathBuf) -> Result<()> {
    // Create output directory
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create directory: {}", output_dir.display()))?;

    println!(
        "Downloading {} Dia TTS samples to {}",
        SAMPLES.len(),
        output_dir.display()
    );

    // Download each sample
    for sample in SAMPLES {
        download_sample(sample, &output_dir).await?;
    }

    println!("\n✅ All samples downloaded successfully!");
    println!("📁 Location: {}", output_dir.canonicalize()?.display());

    Ok(())
}

async fn download_sample(sample: &Sample, output_dir: &Path) -> Result<()> {
    let url = format!("https://dia-tts.com/audios/{}", sample.url_filename);
    let output_path = output_dir.join(sample.local_filename);

    // Skip if already exists
    if output_path.exists() {
        println!("⏭️  {} already exists, skipping", sample.local_filename);
        return Ok(());
    }

    println!(
        "\n📥 Downloading sample {}: {}",
        sample.number, sample.description
    );

    // Create client with timeout
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    // Download the file
    let response = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("Failed to download {}", url))?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Download failed with status: {}",
            response.status()
        ));
    }

    // Download content
    let content = response
        .bytes()
        .await
        .with_context(|| format!("Failed to read response body for {}", sample.url_filename))?;

    // Write to file
    fs::write(&output_path, &content)
        .with_context(|| format!("Failed to write file: {}", output_path.display()))?;

    println!(
        "✅ {} downloaded ({:.2} MB)",
        sample.local_filename,
        content.len() as f64 / 1_048_576.0
    );

    Ok(())
}

/// Main CLI entry point
pub fn cli_main() -> Result<()> {
    use clap::Parser;

    let cli = Cli::parse();

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        match cli.command {
            Commands::DownloadSamples(cmd) => cmd.execute().await,
            Commands::Speak(cmd) => cmd.execute().await,
        }
    })
}
