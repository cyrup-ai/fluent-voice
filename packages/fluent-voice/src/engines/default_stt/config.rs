//! Configuration Structures for Default STT Engine
//!
//! Zero-allocation, stack-based configurations for VAD and wake word detection
//! with compile-time optimizations and performance guarantees.

/// Zero-Allocation VAD Configuration: Stack-based, compile-time optimized
#[derive(Copy, Clone, Debug)]
pub struct VadConfig {
    /// VAD sensitivity threshold (0.0 to 1.0) - stack allocated
    pub sensitivity: f32,
    /// Minimum speech duration in milliseconds - compile-time constant
    pub min_speech_duration: u32,
    /// Maximum silence duration in milliseconds - compile-time constant
    pub max_silence_duration: u32,
    /// SIMD optimization level (0=none, 1=SSE, 2=AVX2, 3=AVX512)
    pub simd_level: u8,
}

impl Default for VadConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            sensitivity: 0.5,
            min_speech_duration: 250,
            max_silence_duration: 1500,
            simd_level: 2, // AVX2 by default for blazing-fast performance
        }
    }
}

/// Zero-Allocation Wake Word Configuration: Stack-based, no string allocations
#[derive(Copy, Clone, Debug)]
pub struct WakeWordConfig {
    /// Wake word model index (0="syrup", 1="hey", 2="ok") - no string allocation
    pub model_index: u8,
    /// Detection sensitivity threshold - stack allocated
    pub sensitivity: f32,
    /// Sub-millisecond detection enabled
    pub ultra_low_latency: bool,
    /// Enable audio filters for noise reduction
    pub filters_enabled: bool,
    /// Band-pass filter configuration
    pub band_pass_enabled: bool,
    pub band_pass_low_cutoff: f32,
    pub band_pass_high_cutoff: f32,
    /// Gain normalizer configuration
    pub gain_normalizer_enabled: bool,
    pub gain_normalizer_max_gain: f32,
}

impl Default for WakeWordConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            model_index: 0, // "syrup" model
            sensitivity: 0.8,
            ultra_low_latency: true,
            // Default noise reduction settings (low level)
            filters_enabled: true,
            band_pass_enabled: true,
            band_pass_low_cutoff: 85.0,
            band_pass_high_cutoff: 8000.0,
            gain_normalizer_enabled: true,
            gain_normalizer_max_gain: 2.0,
        }
    }
}
