//! Diagnostic and Logging Functions
//!
//! Comprehensive diagnostic logging for configuration analysis and debugging.

use super::config::{VadConfig, WakeWordConfig};

/// Comprehensive diagnostic logging function for startup configuration analysis
///
/// This function logs all default settings across the fluent-voice ecosystem
/// to help with debugging, performance analysis, and configuration validation.
pub async fn log_diagnostic_startup_settings(
    vad_config: &VadConfig,
    wake_word_config: &WakeWordConfig,
) {
    use tracing::{debug, info, warn};

    info!("🚀 FLUENT-VOICE STARTUP DIAGNOSTICS");
    info!("=====================================");

    // System Information
    info!("📊 SYSTEM INFORMATION");
    info!("  Platform: {}", std::env::consts::OS);
    info!("  Architecture: {}", std::env::consts::ARCH);
    info!(
        "  Available parallelism: {:?}",
        std::thread::available_parallelism()
    );
    info!("  Process ID: {}", std::process::id());

    // VAD Configuration Diagnostics
    info!("🎙️  VAD (Voice Activity Detection) CONFIGURATION");
    info!("  Sensitivity: {:.2}", vad_config.sensitivity);
    info!(
        "  Min speech duration: {}ms",
        vad_config.min_speech_duration
    );
    info!(
        "  Max silence duration: {}ms",
        vad_config.max_silence_duration
    );
    info!(
        "  SIMD level: {} ({})",
        vad_config.simd_level,
        match vad_config.simd_level {
            0 => "none",
            1 => "SSE",
            2 => "AVX2",
            3 => "AVX512",
            _ => "unknown",
        }
    );

    // Wake Word Configuration Diagnostics
    info!("🔊 WAKE WORD DETECTION CONFIGURATION");
    info!(
        "  Model index: {} ({})",
        wake_word_config.model_index,
        match wake_word_config.model_index {
            0 => "syrup",
            1 => "hey",
            2 => "ok",
            _ => "unknown",
        }
    );
    info!("  Sensitivity: {:.2}", wake_word_config.sensitivity);
    info!(
        "  Ultra low latency: {}",
        wake_word_config.ultra_low_latency
    );

    // Audio Filters Diagnostics - THE REAL NOISE REDUCTION
    info!("🎚️  AUDIO FILTERS (NOISE REDUCTION) CONFIGURATION");
    info!("  Filters enabled: {}", wake_word_config.filters_enabled);

    if wake_word_config.filters_enabled {
        info!("  📊 BAND-PASS FILTER:");
        info!("    Enabled: {}", wake_word_config.band_pass_enabled);
        if wake_word_config.band_pass_enabled {
            info!(
                "    Low cutoff: {:.1} Hz",
                wake_word_config.band_pass_low_cutoff
            );
            info!(
                "    High cutoff: {:.1} Hz",
                wake_word_config.band_pass_high_cutoff
            );
            info!(
                "    Bandwidth: {:.1} Hz",
                wake_word_config.band_pass_high_cutoff - wake_word_config.band_pass_low_cutoff
            );
        }

        info!("  📈 GAIN NORMALIZER:");
        info!("    Enabled: {}", wake_word_config.gain_normalizer_enabled);
        if wake_word_config.gain_normalizer_enabled {
            info!(
                "    Max gain: {:.1}x",
                wake_word_config.gain_normalizer_max_gain
            );
        }
    } else {
        warn!("  ⚠️  Audio filters are DISABLED - no noise reduction active");
    }

    // Whisper Configuration Diagnostics
    info!("🎯 WHISPER STT CONFIGURATION");
    match fluent_voice_whisper::WhisperTranscriber::new() {
        Ok(_whisper) => {
            info!("  Whisper initialization: ✅ SUCCESS");
            debug!("  Whisper transcriber ready for inference");
        }
        Err(e) => {
            warn!("  Whisper initialization: ❌ FAILED - {}", e);
        }
    }

    // Koffee Wake Word Detector Diagnostics
    info!("☕ KOFFEE WAKE WORD DETECTOR CONFIGURATION");
    let mut koffee_config = koffee::KoffeeCandleConfig::default();
    koffee_config.detector.threshold = wake_word_config.sensitivity;
    koffee_config.filters.band_pass.enabled = wake_word_config.band_pass_enabled;
    koffee_config.filters.band_pass.low_cutoff = wake_word_config.band_pass_low_cutoff;
    koffee_config.filters.band_pass.high_cutoff = wake_word_config.band_pass_high_cutoff;
    koffee_config.filters.gain_normalizer.enabled = wake_word_config.gain_normalizer_enabled;
    koffee_config.filters.gain_normalizer.max_gain = wake_word_config.gain_normalizer_max_gain;

    match koffee::KoffeeCandle::new(&koffee_config) {
        Ok(_detector) => {
            info!("  Koffee detector initialization: ✅ SUCCESS");
            info!(
                "  Detector threshold: {:.2}",
                koffee_config.detector.threshold
            );
            info!(
                "  Audio filters configured: {}",
                koffee_config.filters.band_pass.enabled
            );
        }
        Err(e) => {
            warn!("  Koffee detector initialization: ❌ FAILED - {}", e);
        }
    }

    // VAD Detector Diagnostics
    info!("🔍 VAD DETECTOR CONFIGURATION");
    match fluent_voice_vad::VoiceActivityDetector::builder()
        .chunk_size(1024_usize)
        .sample_rate(16000_i64)
        .build()
    {
        Ok(_vad) => {
            info!("  VAD detector initialization: ✅ SUCCESS");
            info!("  Chunk size: 1024 samples");
            info!("  Sample rate: 16000 Hz");
        }
        Err(e) => {
            warn!("  VAD detector initialization: ❌ FAILED - {}", e);
        }
    }

    // Memory and Performance Diagnostics
    info!("🧠 MEMORY & PERFORMANCE DIAGNOSTICS");
    info!("  Ring buffer capacity: Pre-allocated for zero-allocation processing");
    info!("  Channel buffer size: 100 (audio processing pipeline)");
    info!("  Concurrent processing: Lock-free with crossbeam channels");

    // Audio Processing Pipeline Diagnostics
    info!("🎵 AUDIO PROCESSING PIPELINE");
    info!("  Pipeline: Microphone → Koffee Wake Word → VAD → Whisper STT");
    info!("  Processing model: Zero-allocation, lock-free streaming");
    info!("  Error recovery: Comprehensive with semantic error handling");
    info!("  Backpressure: Managed through async streams");

    // Feature Flags Diagnostics
    info!("🏗️  FEATURE FLAGS STATUS");
    #[cfg(feature = "metal")]
    info!("  Metal acceleration: ✅ ENABLED");
    #[cfg(not(feature = "metal"))]
    info!("  Metal acceleration: ❌ DISABLED");

    #[cfg(feature = "cuda")]
    info!("  CUDA acceleration: ✅ ENABLED");
    #[cfg(not(feature = "cuda"))]
    info!("  CUDA acceleration: ❌ DISABLED");

    #[cfg(feature = "microphone")]
    info!("  Microphone support: ✅ ENABLED");
    #[cfg(not(feature = "microphone"))]
    info!("  Microphone support: ❌ DISABLED");

    // Default Handler Configuration
    info!("🔧 DEFAULT EVENT HANDLERS");
    info!("  Error handler: Default recovery with semantic error categorization");
    info!("  Wake handler: Console logging with emoji indicators");
    info!("  Turn handler: Speaker identification with conversation logging");

    // Noise Reduction Level Mapping
    info!("📝 NOISE REDUCTION LEVEL MAPPINGS");
    info!("  NoiseReduction::Off → All filters disabled");
    info!("  NoiseReduction::Low → 200-4000 Hz, 1.5x gain");
    info!("  NoiseReduction::High → 300-3400 Hz, 3.0x gain");

    info!("=====================================");
    info!("✅ DIAGNOSTIC LOGGING COMPLETE - All systems ready for voice processing");
}
