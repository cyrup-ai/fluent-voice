//! Setup module for dia
//!
//! This module handles initialization and setup for the dia crate with a focus on:
//! - Zero allocation where possible
//! - Blazing-fast performance
//! - Elegant error handling
//! - Thread safety without locks
//! - Automatic model downloading via progresshub (now with deterministic client construction)

use progresshub::{ProgressHub, ZeroOneOrMany};
use std::path::PathBuf;
use tracing::info;

/// Model file paths structure
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub weights: PathBuf,
    pub tokenizer: PathBuf,
    pub encodec_weights: PathBuf,
}

/// Setup Dia model with automatic download support
pub async fn setup() -> Result<ModelPaths, String> {
    let dia_model_id = "nari-labs/Dia-1.6B";
    let encodec_model_id = "facebook/encodec_24khz";

    // Download Dia model
    info!("Setting up Dia model {dia_model_id}...");
    tracing::info!(model = %dia_model_id, "📥 Starting model download");

    let dia_results = ProgressHub::builder()
        .model(dia_model_id)
        .with_cli_progress()
        .download()
        .await
        .map_err(|e| format!("Failed to download Dia model: {e}"))?;

    // Extract the first DownloadResult from OneOrMany
    let dia_result = dia_results
        .into_iter()
        .next()
        .ok_or_else(|| "No Dia download results returned".to_string())?;

    let dia_model_result = match dia_result.models {
        ZeroOneOrMany::Zero => return Err("No Dia models downloaded".to_string()),
        ZeroOneOrMany::One(model) => model,
        ZeroOneOrMany::Many(mut models) => models
            .pop()
            .ok_or_else(|| "No Dia models in result".to_string())?,
    };

    // Extract Dia model file paths from the download result
    let weights_path = dia_model_result
        .files
        .iter()
        .find(|f| f.filename == "dia-v0_1.pth")
        .ok_or_else(|| "dia-v0_1.pth file not found in download result".to_string())?
        .path
        .clone();

    let tokenizer_path = dia_model_result
        .files
        .iter()
        .find(|f| f.filename == "dia-v0_1")
        .ok_or_else(|| "dia-v0_1 file not found in download result".to_string())?
        .path
        .clone();

    // Download EnCodec model
    info!("Setting up EnCodec model {}...", encodec_model_id);
    tracing::info!(model = %encodec_model_id, "📥 Starting EnCodec model download");

    let encodec_results = ProgressHub::builder()
        .model(encodec_model_id)
        .with_cli_progress()
        .download()
        .await
        .map_err(|e| format!("Failed to download EnCodec model: {e}"))?;

    // Extract the first DownloadResult from OneOrMany
    let encodec_result = encodec_results
        .into_iter()
        .next()
        .ok_or_else(|| "No EnCodec download results returned".to_string())?;

    let encodec_model_result = match encodec_result.models {
        ZeroOneOrMany::Zero => return Err("No EnCodec models downloaded".to_string()),
        ZeroOneOrMany::One(model) => model,
        ZeroOneOrMany::Many(mut models) => models
            .pop()
            .ok_or_else(|| "No EnCodec models in result".to_string())?,
    };

    // Extract EnCodec weights path from the download result
    let encodec_weights_path = encodec_model_result
        .files
        .iter()
        .find(|f| f.filename == "model.safetensors")
        .ok_or_else(|| "EnCodec model.safetensors file not found in download result".to_string())?
        .path
        .clone();

    info!("Successfully set up model files:");
    info!("  Dia Weights: {}", weights_path.display());
    info!("  Tokenizer: {}", tokenizer_path.display());
    info!("  EnCodec Weights: {}", encodec_weights_path.display());
    info!(
        "  Total size: {} bytes",
        dia_result.total_downloaded_bytes + encodec_result.total_downloaded_bytes
    );
    info!(
        "  Download duration: {:.2}s",
        dia_result.total_duration.as_secs_f64() + encodec_result.total_duration.as_secs_f64()
    );

    Ok(ModelPaths {
        weights: weights_path,
        tokenizer: tokenizer_path,
        encodec_weights: encodec_weights_path,
    })
}

/// Configure the dia system with default settings
pub fn configure_defaults() -> Result<(), String> {
    // Initialize tracing - only fails if already initialized, which is fine
    use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

    let _ = tracing_subscriber::registry()
        .with(
            EnvFilter::from_default_env().add_directive(
                "dia=info"
                    .parse()
                    .map_err(|e| format!("Failed to parse tracing directive: {e}"))?,
            ),
        )
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .try_init();

    // Ensure model cache directory exists
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| "Unable to determine cache directory".to_string())?
        .join("dia")
        .join("models");

    std::fs::create_dir_all(&cache_dir).map_err(|e| {
        format!(
            "Failed to create model cache directory {}: {e}",
            cache_dir.display()
        )
    })?;

    // Verify system has sufficient resources
    let available_memory = get_available_memory()?;
    const MIN_MEMORY_GB: u64 = 4;
    if available_memory < MIN_MEMORY_GB * 1024 * 1024 * 1024 {
        return Err(format!(
            "Insufficient system memory. Required: {}GB, Available: {:.2}GB",
            MIN_MEMORY_GB,
            available_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
    }

    // Configure thread pool for model operations
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // Set reasonable defaults based on system capabilities
    unsafe {
        std::env::set_var("DIA_WORKER_THREADS", num_cpus.to_string());
        std::env::set_var("DIA_CACHE_DIR", cache_dir.to_string_lossy().to_string());
    }

    info!("Dia system configured with defaults:");
    info!("  Cache directory: {}", cache_dir.display());
    info!("  Worker threads: {}", num_cpus);
    info!(
        "  Available memory: {:.2}GB",
        available_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    Ok(())
}

/// Get available system memory in bytes
fn get_available_memory() -> Result<u64, String> {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let output = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .map_err(|e| format!("Failed to get system memory: {e}"))?;

        let memory_str =
            String::from_utf8(output.stdout).map_err(|e| format!("Invalid memory output: {e}"))?;

        memory_str
            .trim()
            .parse::<u64>()
            .map_err(|e| format!("Failed to parse memory value: {e}"))
    }

    #[cfg(target_os = "linux")]
    {
        use std::fs;
        let meminfo = fs::read_to_string("/proc/meminfo")
            .map_err(|e| format!("Failed to read /proc/meminfo: {e}"))?;

        for line in meminfo.lines() {
            if line.starts_with("MemAvailable:") {
                let kb = line
                    .split_whitespace()
                    .nth(1)
                    .ok_or_else(|| "Invalid MemAvailable format".to_string())?
                    .parse::<u64>()
                    .map_err(|e| format!("Failed to parse memory: {e}"))?;
                return Ok(kb * 1024); // Convert KB to bytes
            }
        }
        Err("MemAvailable not found in /proc/meminfo".to_string())
    }

    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        use std::mem;

        unsafe {
            let mut mem_status: MEMORYSTATUSEX = mem::zeroed();
            mem_status.dwLength = mem::size_of::<MEMORYSTATUSEX>() as u32;
            
            if GlobalMemoryStatusEx(&mut mem_status) != 0 {
                // Return available physical memory in bytes
                Ok(mem_status.ullAvailPhys)
            } else {
                // Fallback to 8GB if API call fails
                Ok(8 * 1024 * 1024 * 1024)
            }
        }
    }
}
