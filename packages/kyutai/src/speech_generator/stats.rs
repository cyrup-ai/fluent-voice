//! Generation statistics tracking for performance monitoring

use std::sync::atomic::{AtomicUsize, Ordering};

/// Audio sample rate (24kHz - Moshi standard)
const SAMPLE_RATE: u32 = 24000;

/// Statistics for speech generation performance monitoring
#[derive(Debug, Default)]
pub struct GenerationStats {
    /// Total samples generated
    pub samples_generated: AtomicUsize,
    /// Total generation time in milliseconds
    pub generation_time_ms: AtomicUsize,
    /// Number of generation calls
    pub generation_calls: AtomicUsize,
    /// Peak memory usage in bytes
    pub peak_memory_usage: AtomicUsize,
    /// Buffer underruns
    pub buffer_underruns: AtomicUsize,
    /// Audio quality metrics
    pub audio_quality_score: AtomicUsize, // Scaled by 1000 for atomic storage
}

impl GenerationStats {
    /// Get average generation time per sample
    pub fn avg_generation_time_per_sample(&self) -> f64 {
        let total_time = self.generation_time_ms.load(Ordering::Relaxed) as f64;
        let total_samples = self.samples_generated.load(Ordering::Relaxed) as f64;
        if total_samples > 0.0 {
            total_time / total_samples
        } else {
            0.0
        }
    }

    /// Get real-time factor (how much faster than real-time)
    pub fn real_time_factor(&self) -> f64 {
        let total_time = self.generation_time_ms.load(Ordering::Relaxed) as f64;
        let total_samples = self.samples_generated.load(Ordering::Relaxed) as f64;
        if total_time > 0.0 {
            (total_samples / SAMPLE_RATE as f64) / (total_time / 1000.0)
        } else {
            0.0
        }
    }

    /// Get audio quality score (0.0 to 1.0)
    pub fn audio_quality(&self) -> f32 {
        self.audio_quality_score.load(Ordering::Relaxed) as f32 / 1000.0
    }

    /// Record generation metrics
    #[inline]
    pub fn record_generation(&self, samples: usize, time_ms: usize) {
        self.samples_generated.fetch_add(samples, Ordering::Relaxed);
        self.generation_time_ms
            .fetch_add(time_ms, Ordering::Relaxed);
        self.generation_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Record buffer underrun
    #[inline]
    pub fn record_underrun(&self) {
        self.buffer_underruns.fetch_add(1, Ordering::Relaxed);
    }
}
