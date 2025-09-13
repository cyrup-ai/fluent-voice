#! Thin async wrapper around Candle-Whisper that hands out
//! incremental transcripts every `HOP_SECS`.

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use candle_core::{Device, Tensor};
use candle_transformers::models::whisper::{
    Config, DTYPE, HOP_LENGTH, N_FFT, SAMPLE_RATE, SOT_TOKEN, audio, model as w,
};
use crossbeam_channel::{Receiver, Sender};

/// Hop length (sec) for incremental decoding – lower → lower latency.
const HOP_SECS: f32 = 0.5;

/// One message per partial / final segment.
#[derive(Debug, Clone)]
pub enum PartialTranscript {
    /// Interim output, may be revised.
    Interim(String),
    /// A sentence boundary was reached.
    Final(String),
}

/// Convert a token string to token ID using the tokenizer.
fn token_id(tokenizer: &tokenizers::Tokenizer, token: &str) -> anyhow::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => anyhow::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

/// Create mel filter bank for audio processing
fn create_mel_filters(num_mel_bins: usize) -> Vec<f32> {
    let n_fft = N_FFT;
    let sample_rate = SAMPLE_RATE as f32;
    let n_freqs = n_fft / 2 + 1;

    // Mel scale conversion functions
    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).ln();
    let mel_to_hz = |mel: f32| 700.0 * (mel / 2595.0).exp() - 700.0;

    // Create mel-spaced frequency points
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate / 2.0);
    let mel_points: Vec<f32> = (0..=num_mel_bins + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (num_mel_bins + 1) as f32)
        .collect();

    // Convert back to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&mel| mel_to_hz(mel)).collect();

    // Convert to FFT bin indices
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| hz * n_fft as f32 / sample_rate)
        .collect();

    // Create filter bank
    let mut filters = vec![0.0f32; num_mel_bins * n_freqs];

    for m in 0..num_mel_bins {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for k in 0..n_freqs {
            let k_f = k as f32;
            if k_f >= left && k_f <= center {
                filters[m * n_freqs + k] = (k_f - left) / (center - left);
            } else if k_f > center && k_f <= right {
                filters[m * n_freqs + k] = (right - k_f) / (right - center);
            }
        }
    }

    filters
}

/// Kick-off a background thread that receives raw PCM samples *after* VAD
/// deemed them speech-worthy, converts them to mel-spectrogram, and calls
/// Whisper every `HOP_SECS`.
pub fn spawn_whisper_stream(
    cfg: Config,
    model_id: String,
    tokenizer_path: std::path::PathBuf,
    device: Device,
) -> anyhow::Result<(Sender<f32>, Receiver<PartialTranscript>)> {
    let (pcm_tx, pcm_rx) = crossbeam_channel::bounded::<f32>(32_768);
    let (text_tx, text_rx) = crossbeam_channel::bounded::<PartialTranscript>(128);

    tokio::spawn(async move {
        if let Err(e) = whisper_loop(cfg, model_id, tokenizer_path, device, pcm_rx, text_tx).await {
            eprintln!("whisper thread crashed: {e:?}");
        }
    });

    Ok((pcm_tx, text_rx))
}

async fn whisper_loop(
    cfg: Config,
    model_id: String,
    tokenizer: std::path::PathBuf,
    device: Device,
    pcm_rx: Receiver<f32>,
    text_tx: Sender<PartialTranscript>,
) -> anyhow::Result<()> {
    use candle_nn::ops::softmax;
    use progresshub::ProgressHub;
    use tokenizers::Tokenizer;

    let tokenizer = Tokenizer::from_file(tokenizer).expect("tokenizer");
    let vb = {
        // Download model using progresshub builder API
        let download_result = ProgressHub::builder()
            .model(&model_id)
            .build()
            .model(&model_id)
            .await?;

        let weights = match &download_result.models {
            progresshub::ZeroOneOrMany::One(model) => model
                .files
                .iter()
                .find(|f| f.filename == "model.safetensors")
                .ok_or_else(|| anyhow::anyhow!("model.safetensors not found"))?
                .path
                .clone(),
            progresshub::ZeroOneOrMany::Many(models) => models
                .first()
                .ok_or_else(|| anyhow::anyhow!("No models in download result"))?
                .files
                .iter()
                .find(|f| f.filename == "model.safetensors")
                .ok_or_else(|| anyhow::anyhow!("model.safetensors not found"))?
                .path
                .clone(),
            progresshub::ZeroOneOrMany::Zero => {
                return Err(anyhow::anyhow!("No models downloaded"));
            }
        };

        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], DTYPE, &device)? }
    };
    let mut model = w::Whisper::load(&vb, cfg.clone())?;

    let mut mel_buf: Vec<f32> = Vec::with_capacity(SAMPLE_RATE * HOP_LENGTH);
    let last_decode = Arc::new(Mutex::new(Instant::now()));

    loop {
        // Block until some samples arrive (mic already pre-filtered by VAD)
        let s = pcm_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("pcm sender hung up"))?;
        mel_buf.push(s);

        // Decode whenever hop interval elapsed
        if last_decode.lock().unwrap().elapsed() < Duration::from_secs_f32(HOP_SECS) {
            continue;
        }
        *last_decode.lock().unwrap() = Instant::now();

        let (mel, mel_len) = {
            // Create mel filters programmatically
            let mel_filters = create_mel_filters(cfg.num_mel_bins);
            let m = audio::pcm_to_mel(&cfg, &mel_buf, &mel_filters);
            let l = m.len() / cfg.num_mel_bins;
            (m, l)
        };
        if mel_len < 4 {
            continue;
        } // not enough yet

        let mel_t = Tensor::from_vec(mel, (1, cfg.num_mel_bins, mel_len), &device)?;
        // basic greedy decode – temperature 0
        let sot_token_id = token_id(&tokenizer, SOT_TOKEN)?;
        let tokens_t = Tensor::new(&[sot_token_id], &device)?.unsqueeze(0)?;
        let feats = model.encoder.forward(&mel_t, true)?;
        let ys = model.decoder.forward(&tokens_t, &feats, true)?;
        let logits = model
            .decoder
            .final_linear(&ys.narrow(0, 0, 1)?)?
            .squeeze(0)?
            .squeeze(0)?;
        let next_id = softmax(&logits, candle_core::D::Minus1)?
            .argmax(candle_core::D::Minus1)?
            .to_scalar::<i64>()? as u32;

        let text = tokenizer
            .decode(&[next_id], true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        if text.contains('\n') || text.ends_with(['.', '!', '?', '。', '！', '？']) {
            let _ = text_tx.send(PartialTranscript::Final(text.trim().into()));
            mel_buf.clear(); // reset for next segment
        } else {
            let _ = text_tx.send(PartialTranscript::Interim(text.trim().into()));
        }
    }
}
