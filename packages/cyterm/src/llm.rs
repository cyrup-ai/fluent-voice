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
    let weights_path = hf_hub::api::sync::Api::new()?
        .repo(hf_hub::Repo::model(cfg.model_id.clone()))
        .get(&cfg.which_bin)?;

    let mut file = std::fs::File::open(weights_path)?;
    let config = m::Config::from_reader(&mut file)?;
    let weights = m::weights::TransformerWeights::from_reader(&mut file, &config, &device)?;
    let vb = weights.var_builder(&config, &device)?;
    let mut cache = m::Cache::new(true, &config, vb.pp("rot"))?;
    let llama = m::Llama::load(vb, config.clone())?;

    let tokenizer = hf_hub::api::sync::Api::new()?
        .model("hf-internal-testing/llama-tokenizer".to_string())
        .get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer)?;

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
