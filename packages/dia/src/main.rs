// src/main.rs
// ─────────────────────────────────────────────────────────────────────────────
// Dia-Voice reference CLI
//
//  ❯ cargo run --release -- --prompt "Hello world!"
//  ❯ cargo run --release -- --prompt "Hi, Dave. What's shakin'?" --out shakin.wav
// ─────────────────────────────────────────────────────────────────────────────

use anyhow::{Context, Result};
use candle_core::IndexOp;
use candle_transformers::generation::LogitsProcessor;
use clap::Parser;
use dia::{DType, Device, Tensor, VarBuilder};
use indicatif::{ProgressBar, ProgressStyle};
use tokenizers::Tokenizer;

use dia::{
    audio::{SAMPLE_RATE, channel_delay, normalize_loudness, play_pcm, write_pcm_as_wav},
    codec::encode_wav, // optional audio-prompt support
    config::DiaConfig,
    model::{DiaModel, set_model_paths},
    setup,
};

// GPU optimizations have been simplified - channel_delay_gpu removed

/// Anything below this RMS is considered silence (skip LUFS stage).
const SILENCE_THRESHOLD: f32 = 1e-4;

/// CLI switches.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Force CPU execution (otherwise CUDA/Metal if available).
    #[arg(long)]
    cpu: bool,

    /// Path to a local `model.safetensors`.  If omitted we pull from HF.
    #[arg(long)]
    weights: Option<String>,

    /// Optional path to `tokenizer.json`.
    #[arg(long)]
    tokenizer: Option<String>,

    /// Optional path to an **audio prompt** (will be encoded with EnCodec).
    #[arg(long)]
    prompt_wav: Option<String>,

    /// Text prompt.
    #[arg(long, default_value = "[SP1] Hey Dave! What's crackalackin? (excited)")]
    prompt: String,

    /// Sampling seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Temperature (0.0 = greedy).
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,

    /// Nucleus‐sampling p (omit for disabled).
    #[arg(long)]
    top_p: Option<f64>,

    /// Output WAV file (if not provided, audio plays through speakers).
    #[arg(long)]
    out: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    async_main().await
}

async fn async_main() -> anyhow::Result<()> {
    // ───────────── CLI & device setup ─────────────────────────────────────
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else if candle_core::utils::cuda_is_available() {
        match Device::new_cuda(0) {
            Ok(device) => {
                tracing::info!("• using CUDA device");
                device
            }
            Err(e) => {
                tracing::warn!(error = %e, "• CUDA error");
                Device::Cpu
            }
        }
    } else if candle_core::utils::metal_is_available() {
        match Device::new_metal(0) {
            Ok(device) => {
                tracing::info!("• using Metal device");
                device
            }
            Err(e) => {
                tracing::warn!(error = %e, "• Metal error, falling back to CPU");
                Device::Cpu
            }
        }
    } else {
        tracing::info!(
            "• no GPU acceleration available (compile with --features cuda or --features metal)"
        );
        Device::Cpu
    };
    tracing::info!(
        device = ?device,
        avx = candle_core::utils::with_avx(),
        neon = candle_core::utils::with_neon(),
        simd128 = candle_core::utils::with_simd128(),
        f16c = candle_core::utils::with_f16c(),
        "• device features"
    );

    // ───────────── Weights / tokenizer download with progresshub CLI ─────────────────
    let (weights, tokenizer_path): (std::path::PathBuf, std::path::PathBuf) =
        if let (Some(weights_path), Some(tokenizer_path)) = (&args.weights, &args.tokenizer) {
            // Both provided locally - no downloads needed
            tracing::info!("• using local model files");
            (
                std::path::PathBuf::from(weights_path),
                std::path::PathBuf::from(tokenizer_path),
            )
        } else {
            tracing::info!("• checking for model files...");
            // Use progresshub's beautiful CLI progress display
            let model_paths = setup::setup()
                .await
                .map_err(|e| anyhow::anyhow!("Setup failed: {}", e))?;

            // Store the model paths globally for load_encodec to use
            set_model_paths(model_paths.clone())
                .map_err(|e| anyhow::anyhow!("Failed to set model paths: {}", e))?;

            // Return the paths from the setup function
            (model_paths.weights, model_paths.tokenizer)
        };

    // ───────────── Dia-Voice model (mmap) ─────────────────────────────────
    let cfg = DiaConfig::default(); // local hard-coded config
    // Use BF16 for CUDA/Metal, F32 otherwise
    let dtype = if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    };
    let vb = if weights.extension().and_then(|s| s.to_str()) == Some("pth") {
        // Load PyTorch model
        VarBuilder::from_pth(weights, dtype, &device)?
    } else {
        // Load SafeTensors model (default)
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)? }
    };
    let dia = DiaModel::new(cfg.clone(), vb, dtype)?; // use appropriate dtype

    // ───────────── Tokenise text prompt ───────────────────────────────────
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    let prompt_ids = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e))?
        .get_ids()
        .to_vec();
    let prompt_ids = Tensor::new(prompt_ids, &device)?.unsqueeze(0)?; // [1,S] → text encoder expects B=2

    // ───────────── Optional audio prompt → EnCodec codes ─────────────────
    let audio_prompt_codes = if let Some(path) = &args.prompt_wav {
        tracing::info!("• encoding audio prompt…");
        Some(encode_wav(path, &device, /*compress*/ true)?)
    } else {
        None
    };

    // ───────────── Encoder forward pass ───────────────────────────────────
    let prompt_b2s = prompt_ids.expand(&[2, prompt_ids.dim(1)?])?; // duplicate for CFG (uncond/cond)
    let (enc_out, enc_state) = dia.encode(&prompt_b2s)?;
    let cross_cache = dia.build_cross_cache(&enc_out, &enc_state.positions)?;

    // ───────────── Prepare decoder state ──────────────────────────────────
    let mut dec_state = dia.new_decoder_state(&enc_state, enc_out, cross_cache, &device)?;

    // If we have an audio prompt, pre-fill the decoder with it (mimic BOS).
    if let Some(ap) = &audio_prompt_codes {
        // Apply temporal delays before prefilling decoder
        let ap_delayed = channel_delay::delayed_view(ap, cfg.data.audio_pad_value)?;

        let ap_b2tc = ap_delayed
            .unsqueeze(0)? // [T,C] → [1,T,C]
            .expand(&[2, ap.dim(0)?, ap.dim(1)?])?;
        dia.prefill_decoder(&ap_b2tc, &mut dec_state)?;
        // The prefill has been completed with the audio prompt dimensions
    }

    // ───────────── Autoregressive sampling loop ───────────────────────────
    let mut sampler = LogitsProcessor::new(args.seed, Some(args.temperature), args.top_p);
    let mut codes = Vec::<u32>::new();

    // Advanced progress bar for token generation
    let token_pb = ProgressBar::new(cfg.data.audio_length as u64);
    let progress_style = ProgressStyle::with_template(
        "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tokens ({eta})",
    )
    .with_context(|| "Failed to create progress bar template")?
    .progress_chars("█▉▊▋▌▍▎▏ ");

    token_pb.set_style(progress_style);

    tracing::info!("Beginning audio generation...");
    for step in 0..cfg.data.audio_length {
        dec_state.prepare_step(step);

        // Build `[B,1,C]` token tensor for this step.
        let toks = if step == 0 && audio_prompt_codes.is_none() {
            // BOS = PAD for all channels.
            Tensor::zeros(&[2, 1, cfg.data.channels], DType::U32, &device)?
        } else {
            // Previous tokens + PAD sentinel at the end.
            let pad = cfg.data.audio_pad_value;
            let mut tmp = Vec::with_capacity(cfg.data.channels);
            tmp.extend_from_slice(&codes);
            tmp.push(pad);
            Tensor::new(tmp, &device)?.reshape((2, 1, cfg.data.channels))?
        };

        // Apply temporal delays before model sees the tokens
        let toks = channel_delay::delayed_view(&toks, cfg.data.audio_pad_value)?;

        // Decoder forward (CFG).
        let logits = dia.decode_step(&toks, &mut dec_state)?;
        // Select conditional row, convert to f32.
        let logits = logits.i((1, .., ..))?.to_dtype(DType::F32)?;
        let next = sampler.sample(&logits)?;

        if next == cfg.data.audio_eos_value {
            tracing::info!(step, "• <eos> reached");
            break;
        }
        codes.push(next);
        token_pb.inc(1);
    }
    token_pb.finish_with_message(format!("Generated {} audio codes", codes.len()));

    // ───────────── EnCodec decode → waveform ──────────────────────────────
    tracing::info!("• decoding audio with EnCodec...");
    let codes_t =
        Tensor::from_slice(&codes, (codes.len(), cfg.data.channels), &device)?.unsqueeze(0)?; // [1,T,C]

    // Remove temporal delays before EnCodec decoding
    let codes_t = channel_delay::undelayed_view(&codes_t, cfg.data.audio_pad_value)?;

    let pcm = dia
        .decode_audio_codes(&codes_t)
        .map_err(|e| anyhow::anyhow!("Audio decoding failed: {}", e))? // [1,1,T]
        .squeeze(0)?
        .squeeze(0)?; // [T]

    // Helper function to calculate RMS
    fn rms(tensor: &Tensor) -> Result<f32> {
        let squared = tensor.sqr()?;
        let mean = squared.mean_all()?;
        Ok(mean.to_scalar::<f32>()?.sqrt())
    }

    // ───────────── Loudness & final WAV write ─────────────────────────────
    tracing::info!("• applying BS.1770 loudness normalization...");

    let pcm = if rms(&pcm)? < SILENCE_THRESHOLD {
        pcm
    } else {
        normalize_loudness(&pcm, SAMPLE_RATE as u32, /*compress*/ true)?
    };
    let pcm_vec = pcm.to_vec1::<f32>()?;

    // ─────────────  Deliver audio  ────────────────────────────────────────
    match &args.out {
        Some(out_path) => {
            // ── Write to WAV file ─────────────────────────────────────────
            tracing::info!(path = %out_path, "• writing WAV...");
            let mut wav_out =
                std::fs::File::create(out_path).with_context(|| format!("create {out_path}"))?;
            write_pcm_as_wav(&mut wav_out, pcm_vec, SAMPLE_RATE as u32, None)?;
            tracing::info!(path = %out_path, "✔ wrote file");
        }
        None => {
            // ── Stream to the default output device ────────────────────────
            tracing::info!("• playing audio through speakers...");
            play_pcm(&pcm_vec, SAMPLE_RATE as u32)?;
            tracing::info!(
                samples = pcm_vec.len(),
                sample_rate = SAMPLE_RATE,
                "✔ playback complete"
            );
        }
    }
    Ok(())
}
