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
use candle_transformers::models::llama2_c_weights;
use progresshub::ProgressHub;
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

    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            if let Err(e) = llm_loop(cfg, device, prompt_rx, reply_tx).await {
                eprintln!("llm thread crashed: {e:?}");
            }
        })
    });

    Ok((prompt_tx, reply_rx))
}



async fn llm_loop(
    cfg: LlmConfig,
    device: Device,
    prompt_rx: Receiver<String>,
    reply_tx: Sender<String>,
) -> anyhow::Result<()> {
    // Download model using progresshub builder API
    let model_download = ProgressHub::builder()
        .model(&cfg.model_id)
        .build()
        .model(&cfg.model_id)
        .await?;

    let weights_path = match &model_download.models {
        progresshub::ZeroOneOrMany::One(model) => model
            .files
            .iter()
            .find(|f| f.filename == cfg.which_bin)
            .ok_or_else(|| anyhow::anyhow!("Model weights file '{}' not found", cfg.which_bin))?
            .path
            .clone(),
        progresshub::ZeroOneOrMany::Many(models) => models
            .first()
            .ok_or_else(|| anyhow::anyhow!("No models in download result"))?
            .files
            .iter()
            .find(|f| f.filename == cfg.which_bin)
            .ok_or_else(|| anyhow::anyhow!("Model weights file '{}' not found", cfg.which_bin))?
            .path
            .clone(),
        progresshub::ZeroOneOrMany::Zero => {
            return Err(anyhow::anyhow!("No models downloaded"));
        }
    };

    let mut file = std::fs::File::open(weights_path)?;
    let config = m::Config::from_reader(&mut file)?;
    let weights = llama2_c_weights::TransformerWeights::from_reader(&mut file, &config, &device)?;
    let vb = weights.var_builder(&config, &device)?;
    let mut cache = m::Cache::new(true, &config, vb.pp("rot"))?;
    let llama = m::Llama::load(vb, config.clone())?;

    let tokenizer_download = ProgressHub::builder()
        .model("hf-internal-testing/llama-tokenizer")
        .build()
        .model("hf-internal-testing/llama-tokenizer")
        .await?;

    let tokenizer_path = match &tokenizer_download.models {
        progresshub::ZeroOneOrMany::One(model) => model
            .files
            .iter()
            .find(|f| f.filename == "tokenizer.json")
            .ok_or_else(|| anyhow::anyhow!("tokenizer.json not found"))?
            .path
            .clone(),
        progresshub::ZeroOneOrMany::Many(models) => models
            .first()
            .ok_or_else(|| anyhow::anyhow!("No tokenizer in download result"))?
            .files
            .iter()
            .find(|f| f.filename == "tokenizer.json")
            .ok_or_else(|| anyhow::anyhow!("tokenizer.json not found"))?
            .path
            .clone(),
        progresshub::ZeroOneOrMany::Zero => {
            return Err(anyhow::anyhow!("No tokenizer downloaded"));
        }
    };
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("{}", e))?;

    loop {
        // block on a sentence from the ASR pipeline
        let sentence = prompt_rx.recv()?;
        let mut ids = tokenizer
            .encode(sentence, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .get_ids()
            .to_vec();
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
            let logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(1)?;
            let next = logits_proc.sample(&logits)?;
            buf.push(next);
            // stream every token
            let token = tokenizer
                .decode(&[next], false)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let _ = reply_tx.send(token);
        }
        thread::sleep(Duration::from_millis(10));
    }
}
