//! Command Line Interface for Koffee Wake Word Engine
//!
//! This module provides a CLI interface for training and using wake word models.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// Koffee Wake Word Engine CLI
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Available commands for the Koffee CLI
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Train a new wake word model
    Train(TrainCommand),

    /// Detect wake words from audio input
    Detect(DetectCommand),

    /// List available audio devices
    ListDevices,

    /// Record training samples
    Record(RecordCommand),

    /// Inspect a trained model
    Inspect(InspectCommand),

    /// Generate synthetic training samples using TTS
    Generate(GenerateCommand),
}

/// Train a new wake word model
#[derive(Parser, Debug)]
pub struct TrainCommand {
    /// Directory containing training data
    #[arg(short, long)]
    pub data_dir: PathBuf,

    /// Output path for the trained model
    #[arg(short, long)]
    pub output: PathBuf,

    /// Model type (tiny, small, medium, large)
    #[arg(short, long, default_value = "small")]
    pub model_type: ModelType,

    /// Learning rate
    #[arg(long, default_value_t = 0.001)]
    pub learning_rate: f64,

    /// Number of training epochs
    #[arg(short, long, default_value_t = 100)]
    pub epochs: usize,

    /// Batch size
    #[arg(short, long, default_value_t = 8)]
    pub batch_size: usize,
}

/// Detect wake words from audio input
#[derive(Parser, Debug)]
pub struct DetectCommand {
    /// Path to the wake word model
    #[arg(short, long)]
    pub model: PathBuf,

    /// Path to the stop word model (optional)
    #[arg(long)]
    pub stop_model: Option<PathBuf>,

    /// Audio device to use (use list-devices to see available devices)
    #[arg(short, long)]
    pub device: Option<String>,

    /// Detection threshold (0.0 to 1.0)
    #[arg(short, long, default_value_t = 0.5)]
    pub threshold: f32,
}

/// Record training samples
#[derive(Parser, Debug)]
pub struct RecordCommand {
    /// Output directory for recorded samples
    #[arg(short, long)]
    pub output_dir: PathBuf,

    /// Label for the samples
    #[arg(short, long)]
    pub label: String,

    /// Number of samples to record
    #[arg(short, long, default_value_t = 5)]
    pub count: u32,

    /// Duration of each sample in seconds
    #[arg(short, long, default_value_t = 5)]
    pub duration: u32,
}

/// Inspect a trained model
#[derive(Parser, Debug)]
pub struct InspectCommand {
    /// Path to the model file to inspect
    pub model_path: PathBuf,
}

/// Generate synthetic training samples using TTS
#[derive(Parser, Debug)]
pub struct GenerateCommand {
    /// Phrase to generate samples for (e.g., "hey koffee")
    #[arg(short, long)]
    pub phrase: String,

    /// Output directory for generated samples
    #[arg(short, long, default_value = "synthetic_samples")]
    pub output_dir: PathBuf,

    /// Number of samples to generate
    #[arg(short, long, default_value_t = 10)]
    pub count: usize,

    /// Voice model to use (path to voice file or directory)
    #[arg(short, long)]
    pub voice_model: Option<PathBuf>,

    /// Voice timber (neutral, warm, bright, dark, rich, soft, clear, strong)
    #[arg(long, default_value = "neutral")]
    pub timber: VoiceTimber,

    /// Voice speed (x-slow, slow, normal, fast, x-fast)
    #[arg(long, default_value = "normal")]
    pub speed: VoiceSpeed,

    /// Apply noise reduction
    #[arg(long, default_value_t = true)]
    pub noise_reduction: bool,

    /// Target loudness in LUFS (default: -16.0)
    #[arg(long, default_value_t = -16.0)]
    pub target_loudness: f32,
}

/// Voice timber options
#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum VoiceTimber {
    Neutral,
    Warm,
    Bright,
    Dark,
    Rich,
    Soft,
    Clear,
    Strong,
}

/// Voice speed options
#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum VoiceSpeed {
    XSlow,
    Slow,
    Normal,
    Fast,
    XFast,
}

/// Supported model types
#[derive(ValueEnum, Clone, Debug)]
pub enum ModelType {
    Tiny,
    Small,
    Medium,
    Large,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Tiny => write!(f, "tiny"),
            ModelType::Small => write!(f, "small"),
            ModelType::Medium => write!(f, "medium"),
            ModelType::Large => write!(f, "large"),
        }
    }
}

impl std::str::FromStr for ModelType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "tiny" => Ok(ModelType::Tiny),
            "small" => Ok(ModelType::Small),
            "medium" => Ok(ModelType::Medium),
            "large" => Ok(ModelType::Large),
            _ => Err(format!("Unknown model type: {s}")),
        }
    }
}
