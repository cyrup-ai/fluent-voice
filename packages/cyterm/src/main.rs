//! cyterm – demo entry-point showing how the wake-word detector is wired in.
//! For real production code you’ll probably split this into modules, but
//! keeping it in one file makes the integration crystal-clear.

use anyhow::{Context, Result};
#[cfg(feature = "microphone")]
use cpal::{
    SampleFormat,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use cyterm::{
    asr::VoiceActivityDetector,
    wake_word::{KwModel, WakeWordBuilder},
};

/// 16-kHz mono, 16-bit signed PCM
const SAMPLE_RATE: u32 = 16_000;
const BLOCK_SIZE: usize = 160; // 10 ms @ 16 kHz

fn main() -> Result<()> {
    // -------------------------------------------------- bootstrap
    init_logging();

    // -------------------------------------------------- load / validate model
    let kw_model = KwModel::load("assets/kw_model.bin").context("opening wake-word model")?;

    // -------------------------------------------------- build VAD
    let vad = VoiceActivityDetector::builder()
        .chunk_size(BLOCK_SIZE)
        .sample_rate(SAMPLE_RATE as i64)
        .build()
        .context("init VAD")?;

    // -------------------------------------------------- compose detector
    let mut detector = WakeWordBuilder::new(kw_model)
        .thresholds(0.55, 0.82) // tweak at runtime
        .build(vad);

    // -------------------------------------------------- open microphone
    let host = cpal::default_host();
    let device = host.default_input_device().context("no input device")?;
    let config = device.default_input_config().context("no default config")?;
    ensure_format(&config)?;

    // ring-buffer for assembling 160-sample blocks
    let mut scratch: Vec<i16> = Vec::with_capacity(BLOCK_SIZE);

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[i16], _| {
            for &sample in data {
                scratch.push(sample);
                if scratch.len() == BLOCK_SIZE {
                    // ---- detector call --------------------------------------
                    if let Ok(true) = detector.push_block(
                        &scratch
                            .iter()
                            .map(|v| *v as f32 / 32_768.0)
                            .collect::<Vec<_>>(),
                    ) {
                        println!("🔊  Wake word detected!");
                    }
                    scratch.clear();
                }
            }
        },
        |err| eprintln!("stream-error: {err}"),
        None,
    )?;
    stream.play()?;

    // Keep the stream alive.
    println!("Listening…   (Ctrl-C to quit)");
    loop {
        std::thread::park();
    }
}

// ------------------------------------------------------------ helpers

fn init_logging() {
    // `RUST_LOG` Env      default level
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
}

/// Bail out if the capture format isn’t 16-kHz/16-bit/mono.
/// Keeps the example simple; in real life you could resample or reformat.
fn ensure_format(cfg: &cpal::SupportedStreamConfig) -> Result<()> {
    if cfg.channels() != 1
        || cfg.sample_rate().0 != SAMPLE_RATE
        || cfg.sample_format() != SampleFormat::I16
    {
        anyhow::bail!(
            "expected 16-bit mono @16 kHz, got {:?} ({} ch, {} Hz)",
            cfg.sample_format(),
            cfg.channels(),
            cfg.sample_rate().0
        );
    }
    Ok(())
}
