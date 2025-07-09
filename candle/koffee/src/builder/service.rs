//! src/wakewords/builder.rs  (new file — compile-time only, not yet wired into CLI)

use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use crate::{Kfc, WakewordLoad, wakewords::WakewordModel};

/// Single‐use builder for a streaming wake-word detector.
///
/// Internally we create a **bounded** channel (so that the audio-thread is
/// never blocked for long) and give the *sender* to the detector while the
/// caller only keeps the *receiver*.
/// ```no_run
/// use koffee_candle::wakewords::Builder;
///
/// let (mut detector, rx) = Builder::new()
///     .with_model("assets/hey_rust_tiny.kc")   // or multiple calls
///     .band_size(5)                            // optional fine-tuning …
///     .build()?;
///
/// // ─ push PCM here ─────────────────────────────────────────────
/// // detector.process_bytes(...);
///
/// // ─ consume events elsewhere (GUI, game-loop, …) ──────────────
/// while let Ok(det) = rx.recv() {
///     println!("🔥 woke on '{}' (score {:.3})", det.name, det.score);
/// }
/// ```
pub struct Builder {
    models: Vec<WakewordModel>,
    band_size: u16,
    score_ref: f32,
    channel_cap: usize,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            band_size: 5,
            score_ref: 0.22,
            channel_cap: 16,
        }
    }
}

impl Builder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load a compiled *.kc* model (can be called multiple times).
    pub fn with_model<P: AsRef<std::path::Path>>(mut self, p: P) -> crate::Result<Self> {
        let m = WakewordModel::load_from_file(p).map_err(|e| e.to_string())?;
        self.models.push(m);
        Ok(self)
    }

    /* ------------ optional fine-tuning ------------ */

    /// Set the DTW band size for matching.
    pub fn band_size(mut self, n: u16) -> Self {
        self.band_size = n;
        self
    }
    /// Set the reference score threshold.
    pub fn score_ref(mut self, r: f32) -> Self {
        self.score_ref = r;
        self
    }
    /// Set the channel buffer capacity.
    pub fn channel_capacity(mut self, c: usize) -> Self {
        self.channel_cap = c;
        self
    }

    /* ------------ finaliser ------------ */

    /// Construct the detector **plus** a [`Receiver`] for wake events.
    pub fn build(self) -> crate::Result<(Kfc, Receiver<crate::KoffeeCandleDetection>)> {
        // 1.  create bounded channel
        let (_tx, rx): (SyncSender<_>, Receiver<_>) = sync_channel(self.channel_cap);

        // 2.  build detector & register every loaded model
        let mut det = Kfc::new(&crate::config::KoffeeCandleConfig::default())?;
        let config = crate::config::DetectorConfig {
            avg_threshold: 0.5,
            threshold: 0.5,
            min_scores: 1,
            eager: false,
            score_ref: self.score_ref,
            band_size: self.band_size,
            score_mode: crate::ScoreMode::Classic,
            vad_mode: None,
            #[cfg(feature = "record")]
            record_path: None,
        };
        det.update_config(&config);

        for m in self.models {
            det.add_wakeword_model(m)?; // new helper (see diff below)
        }

        Ok((det, rx))
    }
}
