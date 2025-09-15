// Whisper CLI - Main binary entry point

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

// Import from our library
use fluent_voice_whisper::{pcm_decode, token_id};
use byteorder;

// Import types that will be re-exported from the library
use candle_transformers::models::whisper::{self as m, Config};

#[cfg(any(feature = "encodec", feature = "mimi", feature = "snac"))]
use candle_transformers::models::whisper::audio;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

// Types that will need to be imported from the library once we restructure
#[cfg(feature = "microphone")]
use fluent_voice_whisper::Model;

#[cfg(not(feature = "microphone"))]
pub enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

#[cfg(not(feature = "microphone"))]
impl Model {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(model) => &model.config,
            Self::Quantized(model) => &model.config,
        }
    }

    pub fn encoder_forward(&mut self, mel: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(model) => model.encoder.forward(mel, flush),
            Self::Quantized(model) => model.encoder.forward(mel, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        tokens: &Tensor,
        audio_features: &Tensor,
        flush: bool,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(model) => model.decoder.forward(tokens, audio_features, flush),
            Self::Quantized(model) => model.decoder.forward(tokens, audio_features, flush),
        }
    }

    pub fn decoder_final_linear(&mut self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(model) => model.decoder.final_linear(xs),
            Self::Quantized(model) => model.decoder.final_linear(xs),
        }
    }
}

fn device_helper(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle_core::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else if candle_core::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else {
        Ok(Device::Cpu)
    }
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
    #[value(name = "distil-large-v3")]
    DistilLargeV3,
}

impl WhichModel {
    pub fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::Large
            | Self::LargeV2
            | Self::LargeV3
            | Self::LargeV3Turbo
            | Self::DistilLargeV2
            | Self::DistilLargeV3 => true,
            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn | Self::DistilMediumEn => {
                false
            }
        }
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
            Self::DistilLargeV3 => ("distil-whisper/distil-large-v3", "main"),
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Task {
    Transcribe,
    Translate,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    model_id: Option<String>,

    /// The model to use, check out available models:
    /// https://huggingface.co/models?search=whisper
    #[arg(long)]
    revision: Option<String>,

    /// The model to be used, can be tiny, small, medium.
    #[arg(long, default_value = "tiny.en")]
    model: WhichModel,

    /// The input to be processed, in wav format, will default to `jfk.wav`. Alternatively
    /// this can be set to sample:jfk, sample:gb1, ... to fetch a sample from the following
    /// repo: https://huggingface.co/datasets/Narsil/candle_demo/
    #[arg(long)]
    input: Option<String>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    quantized: bool,

    /// Language.
    #[arg(long)]
    language: Option<String>,

    /// Task, when no task is specified, the input tokens contain only the sot token which can
    /// improve things when in no-timestamp mode.
    #[arg(long)]
    task: Option<Task>,

    /// Timestamps mode, this is not fully implemented yet.
    #[arg(long)]
    timestamps: bool,

    /// Print the full DecodingResult structure rather than just the text.
    #[arg(long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    let device = device_helper(args.cpu)?;
    let (default_model, default_revision) = if args.quantized {
        ("lmz/candle-whisper", "main")
    } else {
        args.model.model_and_revision()
    };
    let default_model = default_model.to_string();
    let default_revision = default_revision.to_string();
    let (model_id, _revision) = match (args.model_id.clone(), args.revision.clone()) {
        (Some(model_id), Some(revision)) => (model_id, revision),
        (Some(model_id), None) => (model_id, "main".to_string()),
        (None, Some(revision)) => (default_model, revision),
        (None, None) => (default_model, default_revision),
    };

    // Extract values from args before any moves occur to avoid partial move errors
    let input_file = args.input.clone();
    let quantized = args.quantized;
    let model_type = args.model;

    let (config_filename, tokenizer_filename, weights_filename, input) = {
        // Download model using hf-hub
        let api = Api::new()?;
        let repo = api.model(model_id.clone());
        let examples_repo = api.model("Narsil/candle-examples".to_string());

        let sample = if let Some(ref input) = input_file {
            if let Some(sample) = input.strip_prefix("sample:") {
                examples_repo.get(&format!("samples_{sample}.wav"))?
            } else {
                std::path::PathBuf::from(input)
            }
        } else {
            println!("No audio file submitted: Using downloaded samples_jfk.wav");
            examples_repo.get("samples_jfk.wav")?
        };

        let (config, tokenizer, model) = if quantized {
            let ext = match model_type {
                WhichModel::TinyEn => "tiny-en",
                WhichModel::Tiny => "tiny",
                _ => anyhow::bail!("no quantized support for {:?}", model_type),
            };
            (
                repo.get(&format!("config-{ext}.json"))?,
                repo.get(&format!("tokenizer-{ext}.json"))?,
                repo.get(&format!("model-{ext}-q80.gguf"))?,
            )
        } else {
            (
                repo.get("config.json")?,
                repo.get("tokenizer.json")?,
                repo.get("model.safetensors")?,
            )
        };
        (config, tokenizer, model, sample)
    };
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    {
        let (pcm_data, sample_rate) = pcm_decode(input)?;
        if sample_rate != m::SAMPLE_RATE as u32 {
            anyhow::bail!("input file must have a {} sampling rate", m::SAMPLE_RATE)
        }
        println!("pcm data loaded {}", pcm_data.len());
        let mel = audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
            &device,
        )?;
        println!("loaded mel: {:?}", mel.dims());

        process_audio(args, config, device, weights_filename, tokenizer, mel)
    }
}

fn process_audio(
    args: Args,
    config: Config,
    device: Device,
    weights_filename: std::path::PathBuf,
    tokenizer: Tokenizer,
    mel: Tensor,
) -> Result<()> {
    let mut model = if args.quantized {
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &weights_filename,
            &device,
        )?;
        Model::Quantized(m::quantized_model::Whisper::load(&vb, config)?)
    } else {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };
        Model::Normal(m::model::Whisper::load(&vb, config)?)
    };

    let language_token = match (args.model.is_multilingual(), args.language) {
        (true, None) => {
            // Simple language detection fallback - use English token
            match token_id(&tokenizer, "<|en|>") {
                Ok(token_id) => Some(token_id),
                Err(_) => None,
            }
        }
        (false, None) => None,
        (true, Some(language)) => match token_id(&tokenizer, &format!("<|{language}|>")) {
            Ok(token_id) => Some(token_id),
            Err(_) => anyhow::bail!("language {language} is not supported"),
        },
        (false, Some(_)) => {
            anyhow::bail!("a language cannot be set for non-multilingual models")
        }
    };
    
    // For now, we'll need to implement the Decoder in the library
    // This is a temporary solution until we move Decoder to the library
    println!("Whisper CLI setup complete - Decoder implementation needed in library");
    Ok(())
}