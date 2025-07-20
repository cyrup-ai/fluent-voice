//! Minimal streaming interface to Llama-2-C (Candle).

use std::{
    sync::mpsc::{Receiver, Sender, channel},
    thread,
    time::Duration,
};

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "accelerate",
    feature = "mkl"
))]
use candle_core::{Device, Tensor};
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "accelerate",
    feature = "mkl"
))]
use candle_transformers::models::llama2_c as m;
use progresshub_client_selector::{Client, DownloadConfig, Backend};
use tokenizers::Tokenizer;

pub struct LlmConfig {
    pub model_id: String,
    pub which_bin: String,     // e.g. `"stories15M.bin"`
    pub context_tokens: usize, // e.g. 128
    pub temperature: f64,
    pub top_p: f64,
}

pub fn spawn_llm(
    cfg: LlmConfig,
    device: Device,
) -> anyhow::Result<(Sender<String>, Receiver<String>)> {
    let (prompt_tx, prompt_rx) = channel::<String>();
    let (reply_tx, reply_rx) = channel::<String>();

    thread::spawn(move || {
        if let Err(e) = llm_loop(cfg, device, prompt_rx, reply_tx) {
            eprintln!("llm thread crashed: {e:?}");
        }
    });

    Ok((prompt_tx, reply_rx))
}

fn llm_loop(
    cfg: LlmConfig,
    device: Device,
    prompt_rx: Receiver<String>,
    reply_tx: Sender<String>,
) -> anyhow::Result<()> {
    // Download model using real progresshub client
    let client = Client::new(Backend::Auto);
    let model_config = DownloadConfig {
        destination: dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("fluent-voice")
            .join("llama2c")
            .join(&cfg.model_id),
        show_progress: false,
        use_cache: true,
    };

    let model_download = client
        .download_model_auto(&cfg.model_id, &model_config, None)
        .await?;

    let weights_path = model_download
        .models
        .first()
        .ok_or_else(|| anyhow::anyhow!("No models in download result"))?
        .files
        .iter()
        .find(|f| f.path.file_name().and_then(|n| n.to_str()) == Some(&cfg.which_bin))
        .ok_or_else(|| anyhow::anyhow!("Model weights file not found"))?
        .path
        .clone();

    let mut file = std::fs::File::open(weights_path)?;
    let config = m::Config::from_reader(&mut file)?;
    let weights = m::weights::TransformerWeights::from_reader(&mut file, &config, &device)?;
    let vb = weights.var_builder(&config, &device)?;
    let mut cache = m::Cache::new(true, &config, vb.pp("rot"))?;
    let llama = m::Llama::load(vb, config.clone())?;

    let tokenizer_config = DownloadConfig {
        destination: dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("fluent-voice")
            .join("tokenizer"),
        show_progress: false,
        use_cache: true,
    };

    let tokenizer_download = client
        .download_model_auto("hf-internal-testing/llama-tokenizer", &tokenizer_config, None)
        .await?;

    let tokenizer_path = tokenizer_download
        .models
        .first()
        .ok_or_else(|| anyhow::anyhow!("No tokenizer in download result"))?
        .files
        .iter()
        .find(|f| f.path.file_name().and_then(|n| n.to_str()) == Some("tokenizer.json"))
        .ok_or_else(|| anyhow::anyhow!("tokenizer.json not found"))?
        .path
        .clone();
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;

    loop {
        // block on a sentence from the ASR pipeline
        let sentence = prompt_rx.recv()?;
        let mut ids = tokenizer.encode(sentence, true)?.get_ids().to_vec();
        ids.truncate(cfg.context_tokens);

        let mut logits_proc = candle_transformers::generation::LogitsProcessor::new(
            42,
            Some(cfg.temperature),
            Some(cfg.top_p),
        );
        let mut buf = Vec::<u32>::new();
        for idx in 0..32 {
            let ctxt = if idx == 0 {
                &ids
            } else {
                &buf[buf.len() - 1..]
            };
            let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
            let logits = llama.forward(&input, 0, &mut cache)?;
            let logits = logits.i((0, logits.dim(1)? - 1))?;
            let next = logits_proc.sample(&logits)?;
            buf.push(next);
            // stream every token
            let token = tokenizer.decode(&[next], false)?;
            let _ = reply_tx.send(token);
        }
        thread::sleep(Duration::from_millis(10));
    }
}
