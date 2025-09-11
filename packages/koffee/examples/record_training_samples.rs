//! Record Training Samples for Wake Words
//!
//! This utility helps you record high-quality training samples for wake word detection.
//!
//! Usage:
//!   cargo run --example record_training_samples -- --phrase "cyrup stop" --output-dir cyrup_stop_training

use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleRate, StreamConfig};
use hound::{WavSpec, WavWriter};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Parser, Clone)]
#[command(
    author,
    version,
    about = "Record training samples for wake word detection"
)]
struct Args {
    /// The wake word phrase to record
    #[arg(short, long)]
    phrase: String,

    /// Output directory for recordings
    #[arg(short, long, default_value = "training_samples")]
    output_dir: String,

    /// Number of samples to record
    #[arg(short = 'n', long, default_value = "10")]
    num_samples: u32,

    /// Duration of each sample in seconds
    #[arg(short, long, default_value = "2.0")]
    duration: f32,

    /// Record noise samples
    #[arg(long)]
    noise: bool,

    /// Audio visualization during recording
    #[arg(long, default_value = "true")]
    visualize: bool,
}

struct Recorder {
    device: Device,
    config: StreamConfig,
    output_dir: PathBuf,
    phrase: String,
    sample_duration: Duration,
    visualize: bool,
}

impl Recorder {
    fn new(args: Args) -> Result<Self, Box<dyn std::error::Error>> {
        // Create output directory
        let output_dir = PathBuf::from(&args.output_dir);
        fs::create_dir_all(&output_dir)?;

        // Audio setup
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("No default input device available")?;

        let config = StreamConfig {
            channels: 1,
            sample_rate: SampleRate(16000),
            buffer_size: cpal::BufferSize::Default,
        };

        Ok(Self {
            device,
            config,
            output_dir,
            phrase: args.phrase,
            sample_duration: Duration::from_secs_f32(args.duration),
            visualize: args.visualize,
        })
    }

    fn record_sample(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let path = self.output_dir.join(filename);

        println!("\n📝 Recording: {}", filename);
        println!("   Press ENTER when ready to start...");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        // Recording state
        let samples = Arc::new(Mutex::new(Vec::new()));
        let recording = Arc::new(AtomicBool::new(true));
        let level = Arc::new(AtomicU32::new(0));

        // Audio callback
        let samples_clone = Arc::clone(&samples);
        let recording_clone = Arc::clone(&recording);
        let level_clone = Arc::clone(&level);
        let visualize = self.visualize;

        let data_callback = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if !recording_clone.load(Ordering::Relaxed) {
                return;
            }

            // Store samples
            if let Ok(mut s) = samples_clone.lock() {
                s.extend_from_slice(data);
            }

            // Calculate and store audio level for visualization
            if visualize {
                let rms = (data.iter().map(|s| s * s).sum::<f32>() / data.len() as f32).sqrt();
                let level_u32 = (rms * 1000.0) as u32;
                level_clone.store(level_u32, Ordering::Relaxed);
            }
        };

        let error_callback = |err| {
            eprintln!("Audio error: {}", err);
        };

        let stream =
            self.device
                .build_input_stream(&self.config, data_callback, error_callback, None)?;

        // Countdown
        for i in (1..=3).rev() {
            println!("   Starting in {}...", i);
            std::thread::sleep(Duration::from_secs(1));
        }

        println!("🔴 RECORDING - Say '{}' clearly!", self.phrase);
        stream.play()?;

        // Recording with visualization
        let start = std::time::Instant::now();
        while start.elapsed() < self.sample_duration {
            if self.visualize {
                let current_level = level.load(Ordering::Relaxed) as f32 / 1000.0;
                let bar_length = (current_level * 50.0).min(50.0) as usize;
                let bar = "█".repeat(bar_length) + &"░".repeat(50 - bar_length);
                print!("\r   Level: [{}] {:.3}  ", bar, current_level);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            std::thread::sleep(Duration::from_millis(50));
        }

        recording.store(false, Ordering::Relaxed);
        drop(stream);

        if self.visualize {
            println!("\r   Level: [{}] 0.000  ", "░".repeat(50));
        }
        println!("✅ Recording complete!");

        // Save WAV file
        let samples = samples.lock().unwrap();
        self.save_wav(&path, &samples)?;

        println!("💾 Saved to: {}", path.display());

        Ok(())
    }

    fn save_wav(&self, path: &Path, samples: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = WavWriter::create(path, spec)?;

        for &sample in samples {
            let sample_i16 =
                (sample * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            writer.write_sample(sample_i16)?;
        }

        writer.finalize()?;
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("🎤 Wake Word Training Sample Recorder");
    println!("====================================");
    println!();
    println!("📋 Configuration:");
    println!("   Phrase: '{}'", args.phrase);
    println!("   Output: {}", args.output_dir);
    println!("   Samples: {}", args.num_samples);
    println!("   Duration: {}s each", args.duration);
    println!();

    let recorder = Recorder::new(args.clone())?;

    println!("🎙️  Using: {}", recorder.device.name()?);
    println!();

    if args.noise {
        // Record noise samples
        println!("📢 Recording NOISE samples (background/silence)");
        println!("   Do NOT speak during these recordings!");

        for i in 0..args.num_samples.min(5) {
            let filename = format!("noise{}.wav", i);
            recorder.record_sample(&filename)?;

            println!();
            println!("   Progress: {}/{}", i + 1, args.num_samples.min(5));
        }
    } else {
        // Record wake word samples
        let label = args.phrase.replace(" ", "_");

        println!("📢 Recording '{}' samples", args.phrase);
        println!("   Speak clearly and consistently!");

        for i in 0..args.num_samples {
            let filename = format!("{}_{}[{}].wav", label, format!("{:02}", i), args.phrase);

            recorder.record_sample(&filename)?;

            println!();
            println!("   Progress: {}/{}", i + 1, args.num_samples);

            if i < args.num_samples - 1 {
                println!("   Take a short break...");
                std::thread::sleep(Duration::from_secs(2));
            }
        }
    }

    println!();
    println!("🎉 Recording session complete!");
    println!();
    println!("📁 Files saved to: {}", args.output_dir);

    if !args.noise {
        println!();
        println!("💡 Next steps:");
        println!(
            "   1. Record noise samples: cargo run --example record_training_samples -- --noise -n 3"
        );
        println!("   2. Train the model: cargo run --example train_cyrup_stop");
    }

    Ok(())
}
