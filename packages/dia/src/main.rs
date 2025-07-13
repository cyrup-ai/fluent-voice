// src/main.rs
// ─────────────────────────────────────────────────────────────────────────────
// Dia-Voice reference CLI
//
//  ❯ cargo run --release -- --prompt "Hello world!"
//  ❯ cargo run --release -- --prompt "Hi, Dave. What's shakin'?" --out shakin.wav
// ─────────────────────────────────────────────────────────────────────────────

use anyhow::{Context, Result};
#[cfg(any(feature = "cuda", feature = "metal", feature = "accelerate", feature = "mkl"))]
use candle_core::{DType, Device, IndexOp, Tensor};
#[cfg(any(feature = "cuda", feature = "metal", feature = "accelerate", feature = "mkl"))]
use candle_nn::VarBuilder;
#[cfg(any(feature = "cuda", feature = "metal", feature = "accelerate", feature = "mkl"))]
use candle_transformers::generation::LogitsProcessor;
use clap::Parser;
use crossterm::{event, execute, terminal};
use indicatif::{ProgressBar, ProgressStyle};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::{
    io,
    sync::mpsc::{self, Receiver},
    time::Duration,
};
use tokenizers::Tokenizer;
use tokio;

use crate::{
    app::{App, ProgressUpdate},
    audio::{SAMPLE_RATE, channel_delay, normalize_loudness, play_pcm, write_pcm_as_wav},
    codec::encode_wav, // optional audio-prompt support
    config::DiaConfig,
    model::DiaModel,
    setup,
};

// Import optimizations when GPU features are enabled
#[cfg(any(feature = "cuda", feature = "metal"))]
use dia_voice::optimizations::{
    benchmark::{Timer, log_gpu_memory},
    channel_delay_gpu,
};

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

fn main() -> Result<()> {
    // Run the async main function in a tokio runtime
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async_main())
}

async fn async_main() -> anyhow::Result<()> {
    // ───────────── CLI & device setup ─────────────────────────────────────
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        if candle_core::utils::cuda_is_available() {
            match Device::new_cuda(0) {
                Ok(device) => {
                    println!("• using CUDA device");
                    device
                }
                Err(e) => {
                    println!("• CUDA error: {}", e);
                    Device::Cpu
                }
            }
        } else if candle_core::utils::metal_is_available() {
            match Device::new_metal(0) {
                Ok(device) => {
                    println!("• using Metal device");
                    device
                }
                Err(e) => {
                    println!("• Metal error: {}", e);
                    println!("• falling back to CPU");
                    Device::Cpu
                }
            }
        } else {
            println!(
                "• no GPU acceleration available (compile with --features cuda or --features metal)"
            );
            Device::Cpu
        }
    };
    println!(
        "• device = {device:?}   avx:{} neon:{} simd128:{} f16c:{}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c(),
    );

    // ───────────── Weights / tokenizer download with TUI progress ─────────────────
    let (weights, tokenizer_path): (std::path::PathBuf, std::path::PathBuf) =
        if args.weights.is_some() && args.tokenizer.is_some() {
            // Both provided locally - no downloads needed
            println!("• using local model files");
            (
                std::path::PathBuf::from(args.weights.unwrap()),
                std::path::PathBuf::from(args.tokenizer.unwrap()),
            )
        } else {
            println!("• checking for model files...");
            // Need to download one or both files
            // Channel for progress events
            let (tx, rx) = mpsc::channel::<ProgressUpdate>();

            // Setup terminal for TUI
            println!("• initializing download UI...");
            terminal::enable_raw_mode()?;
            let mut stdout = io::stdout();
            execute!(
                stdout,
                terminal::EnterAlternateScreen,
                event::EnableMouseCapture
            )?;
            let backend = CrosstermBackend::new(stdout);
            let mut terminal = Terminal::new(backend)?;

            // Spawn download tasks
            let weights_path = args.weights.clone();
            let tokenizer_path_arg = args.tokenizer.clone();

            // Use the setup function to ensure models are downloaded
            let download_handle = tokio::spawn(async move {
                // Delegate model downloads to the setup module
                let model_paths =
                    setup::setup(weights_path, tokenizer_path_arg, tx.clone()).await?;

                // Set environment variable for EnCodec path
                unsafe {
                    std::env::set_var(
                        "ENCODEC_WEIGHTS_PATH",
                        model_paths.encodec.to_string_lossy().to_string(),
                    );
                }

                // Return the model paths
                Ok::<setup::ModelPaths, anyhow::Error>(model_paths)
            });

            // Run TUI progress loop
            let tui_result = tui_loop(&mut terminal, rx);

            // Cleanup terminal
            terminal::disable_raw_mode()?;
            execute!(
                terminal.backend_mut(),
                terminal::LeaveAlternateScreen,
                event::DisableMouseCapture
            )?;
            terminal.show_cursor()?;

            // Handle any TUI errors
            tui_result?;

            // Wait for downloads to complete and get the model paths
            let model_paths = download_handle.await??;

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
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)? };
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
        println!("• encoding audio prompt…");
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
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let ap_delayed = if matches!(device, Device::Cuda(_) | Device::Metal(_)) {
            channel_delay_gpu::delayed_view_gpu(&ap, cfg.data.audio_pad_value)?
        } else {
            channel_delay::delayed_view(&ap, cfg.data.audio_pad_value)?
        };

        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let ap_delayed = channel_delay::delayed_view(&ap, cfg.data.audio_pad_value)?;

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
    token_pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tokens ({eta})",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏ "),
    );

    println!("Beginning audio generation...");
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
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let toks = if matches!(device, Device::Cuda(_) | Device::Metal(_)) {
            channel_delay_gpu::delayed_view_gpu(&toks, cfg.data.audio_pad_value)?
        } else {
            channel_delay::delayed_view(&toks, cfg.data.audio_pad_value)?
        };

        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let toks = channel_delay::delayed_view(&toks, cfg.data.audio_pad_value)?;

        // Decoder forward (CFG).
        let logits = dia.decode_step(&toks, &mut dec_state)?;
        // Select conditional row, convert to f32.
        let logits = logits.i((1, .., ..))?.to_dtype(DType::F32)?;
        let next = sampler.sample(&logits)?;

        if next == cfg.data.audio_eos_value {
            println!("• <eos> after {step} steps");
            break;
        }
        codes.push(next);
        token_pb.inc(1);
    }
    token_pb.finish_with_message(format!("Generated {} audio codes", codes.len()));

    // ───────────── EnCodec decode → waveform ──────────────────────────────
    println!("• decoding audio with EnCodec...");
    let codes_t =
        Tensor::from_slice(&codes, (codes.len(), cfg.data.channels), &device)?.unsqueeze(0)?; // [1,T,C]

    // Remove temporal delays before EnCodec decoding
    #[cfg(any(feature = "cuda", feature = "metal"))]
    let codes_t = if matches!(device, Device::Cuda(_) | Device::Metal(_)) {
        channel_delay_gpu::undelayed_view_gpu(&codes_t, cfg.data.audio_pad_value)?
    } else {
        channel_delay::undelayed_view(&codes_t, cfg.data.audio_pad_value)?
    };

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let codes_t = channel_delay::undelayed_view(&codes_t, cfg.data.audio_pad_value)?;

    let pcm = dia
        .decode_audio_codes(&codes_t)? // [1,1,T]
        .squeeze(0)?
        .squeeze(0)?; // [T]

    // Helper function to calculate RMS
    fn rms(tensor: &Tensor) -> Result<f32> {
        let squared = tensor.sqr()?;
        let mean = squared.mean_all()?;
        Ok(mean.to_scalar::<f32>()?.sqrt())
    }

    // ───────────── Loudness & final WAV write ─────────────────────────────
    println!("• applying BS.1770 loudness normalization...");

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
            println!("• writing WAV to {}...", out_path);
            let mut wav_out =
                std::fs::File::create(out_path).with_context(|| format!("create {}", out_path))?;
            write_pcm_as_wav(&mut wav_out, pcm_vec, SAMPLE_RATE as u32, None)?;
            println!("✔ wrote {}", out_path);
        }
        None => {
            // ── Stream to the default output device ────────────────────────
            println!("• playing audio through speakers...");
            play_pcm(&pcm_vec, SAMPLE_RATE as u32)?;
            println!("✔ played {} samples at {} Hz", pcm_vec.len(), SAMPLE_RATE);
        }
    }
    Ok(())
}

/// Poll input & progress channel, redraw every 100 ms.
fn tui_loop<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    rx: Receiver<ProgressUpdate>,
) -> io::Result<()> {
    let tick = Duration::from_millis(100);
    let mut app = App::default();

    loop {
        // Drain channel
        while let Ok(p) = rx.try_recv() {
            app.update(p);
        }

        if let Err(e) = terminal.draw(|f| app.draw(f)) {
            eprintln!("Error drawing UI: {}", e);
            break;
        }

        // Handle key presses
        match event::poll(tick) {
            Ok(true) => {
                if let Ok(event::Event::Key(k)) = event::read() {
                    match k.code {
                        event::KeyCode::Char('q') | event::KeyCode::Esc => break,
                        _ => {}
                    }
                }
            }
            Ok(false) => {} // No event, continue
            Err(e) => {
                eprintln!("Error polling events: {}", e);
                break;
            }
        }
    }

    Ok(())
}
