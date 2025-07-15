#! Thin async wrapper around Candle-Whisper that hands out
//! incremental transcripts every `HOP_SECS`.

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use candle_core::{Device, Tensor};
use candle_transformers::models::whisper::{Config, audio, model as w, DTYPE, SOT_TOKEN};
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

/// Kick-off a background thread that receives raw PCM samples *after* VAD
/// deemed them speech-worthy, converts them to mel-spectrogram, and calls
/// Whisper every `HOP_SECS`.
pub fn spawn_whisper_stream(
    cfg: Config,
    tokenizer_path: std::path::PathBuf,
    device: Device,
) -> anyhow::Result<(Sender<f32>, Receiver<PartialTranscript>)> {
    let (pcm_tx, pcm_rx) = crossbeam_channel::bounded::<f32>(32_768);
    let (text_tx, text_rx) = crossbeam_channel::bounded::<PartialTranscript>(128);

    std::thread::spawn(move || {
        if let Err(e) = whisper_loop(cfg, tokenizer_path, device, pcm_rx, text_tx) {
            eprintln!("whisper thread crashed: {e:?}");
        }
    });

    Ok((pcm_tx, text_rx))
}

fn whisper_loop(
    cfg: Config,
    tokenizer: std::path::PathBuf,
    device: Device,
    pcm_rx: Receiver<f32>,
    text_tx: Sender<PartialTranscript>,
) -> anyhow::Result<()> {
    use candle_nn::ops::softmax;
    use hf_hub::api::sync::Api;
    use tokenizers::Tokenizer;

    let tokenizer = Tokenizer::from_file(tokenizer).expect("tokenizer");
    let vb = {
        let weights = Api::new()?
            .repo(cfg.model_id.clone())
            .with_revision("main".to_owned())
            .get("model.safetensors")?;
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], DTYPE, &device)? }
    };
    let mut model = w::Whisper::load(&vb, cfg.clone())?;

    let mut mel_buf: Vec<f32> = Vec::with_capacity(cfg.sample_rate as usize * cfg.hop_length);
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
            let m = audio::pcm_to_mel(&cfg, &mel_buf, &audio::build_mel_filters(&cfg)?);
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
        let logits = model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
        let next_id = softmax(&logits, candle_core::D::Minus1)?
            .argmax(None)?
            .to_scalar::<i64>()? as u32;

        let text = tokenizer.decode(&[next_id], true)?;
        if text.contains('\n') || text.ends_with(['.', '!', '?', '。', '！', '？']) {
            let _ = text_tx.send(PartialTranscript::Final(text.trim().into()));
            mel_buf.clear(); // reset for next segment
        } else {
            let _ = text_tx.send(PartialTranscript::Interim(text.trim().into()));
        }
    }
}
