//! Koffee-Candle command-line interface
//!
//! $ kfc listen --port 13345
//! $ kfc train  --in data/ --out model.kc

use clap::{Parser, Subcommand};
use koffee::config::{DetectorConfig, FiltersConfig, KoffeeCandleConfig};
use koffee::wakewords::{WakewordLoad, WakewordModel};
use koffee::{Kfc, ModelType};

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the wake-word detector (TCP stream of scores on --port)
    Listen {
        #[arg(long, default_value_t = 13345)]
        port: u16,
        /// Path to a compiled *.kc* model
        #[arg(long)]
        model: String,
    },
    /// Train a new model from a directory of wav files
    Train {
        /// Directory with label-encoded wavs
        #[arg(long)]
        input: String,
        /// Output path for *.kc* model
        #[arg(long)]
        output: String,
        #[arg(long, default_value_t = ModelType::Small)]
        model_type: ModelType,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Listen { port, model } => {
            // ---------- build detector -------------------------------------------------
            let model: WakewordModel = WakewordModel::load_file(&model)?;
            let cfg = KoffeeCandleConfig {
                detector: DetectorConfig::default(),
                filters: FiltersConfig::default(),
                fmt: koffee::config::AudioFmt::default(),
            };

            let mut det = Kfc::new(&cfg)?;
            det.add_wakeword("default", model, cfg.detector.score_ref)?;
            // ---------- serve stream ---------------------------------------------------
            koffee::server::run_tcp(&mut det, port)?;
        }
        Cmd::Train {
            input,
            output,
            model_type,
        } => {
            koffee::trainer::train_dir(&input, &output, model_type)?;
        }
    };
    Ok(())
}
