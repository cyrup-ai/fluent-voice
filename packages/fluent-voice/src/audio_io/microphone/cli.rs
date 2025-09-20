use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Task {
    Transcribe,
    Translate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum WhichModel {
    Tiny,
    #[value(name = "tiny.en")]
    TinyEn,
    Base,
    #[value(name = "base.en")]
    BaseEn,
    Small,
    #[value(name = "small.en")]
    SmallEn,
    Medium,
    #[value(name = "medium.en")]
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
    #[value(name = "distil-medium.en")]
    DistilMediumEn,
    #[value(name = "distil-large-v2")]
    DistilLargeV2,
}

impl WhichModel {
    #[allow(dead_code)] // Method used when model loading is re-enabled (line 738 early return currently prevents usage)
    pub fn is_multilingual(&self) -> bool {
        matches!(
            self,
            Self::Tiny
                | Self::Base
                | Self::Small
                | Self::Medium
                | Self::Large
                | Self::LargeV2
                | Self::LargeV3
                | Self::LargeV3Turbo
                | Self::DistilLargeV2
        )
    }

    pub fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::Large => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::LargeV3 => ("openai/whisper-large-v3", "main"),
            Self::LargeV3Turbo => ("openai/whisper-large-v3-turbo", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than GPU.
    #[arg(long)]
    pub cpu: bool,

    #[arg(long)]
    pub model_id: Option<String>,

    /// The model revision to use.
    #[arg(long)]
    pub revision: Option<String>,

    /// The model to use.
    #[arg(long, default_value = "tiny.en")]
    pub model: WhichModel,

    /// Seed for sampling.
    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,

    /// Use quantized models.
    #[arg(long)]
    pub quantized: bool,

    /// Language code.
    #[arg(long)]
    pub language: Option<String>,

    /// Task to perform.
    #[arg(long)]
    pub task: Option<Task>,

    /// Enable timestamps mode.
    #[arg(long)]
    pub timestamps: bool,

    /// Verbose output.
    #[arg(long)]
    pub verbose: bool,

    /// Input device name.
    #[arg(long)]
    pub device: Option<String>,

    /// List available input devices.
    #[arg(long)]
    pub list_devices: bool,
}
