use crate::asr::VoiceActivityDetector;
use crate::features::{FRAME, RING_SIZE, extract as features};
use log::{debug, trace};

pub mod config;
pub mod error;
pub mod model;

pub use config::Config;
pub use error::WakeWordError;
pub use model::KwModel;

pub struct WakeWordDetector {
    model: KwModel,
    vad: VoiceActivityDetector,
    ring: Vec<f32>,
    pos: usize,
    cfg: Config,
}

pub struct Builder {
    model: KwModel,
    cfg: Config,
}

impl Builder {
    pub fn new(model: KwModel) -> Self {
        Self {
            model,
            cfg: Config::default(),
        }
    }

    pub fn thresholds(mut self, vad: f32, wake: f32) -> Self {
        self.cfg.vad_threshold = vad;
        self.cfg.wake_threshold = wake;
        self
    }

    pub fn build(self, vad: VoiceActivityDetector) -> WakeWordDetector {
        WakeWordDetector {
            model: self.model,
            vad,
            ring: vec![0.0; RING_SIZE],
            pos: 0,
            cfg: self.cfg,
        }
    }
}

impl WakeWordDetector {
    pub fn push_block(&mut self, chunk: &[f32]) -> Result<bool, WakeWordError> {
        // ---- sanity check -----------------------------------------------------
        if chunk.len() != FRAME {
            return Err(WakeWordError::WrongChunk {
                expected: FRAME,
                got: chunk.len(),
            });
        }

        // ---- ring-buffer boundaries ------------------------------------------
        if self.pos + FRAME > self.ring.len() {
            return Err(WakeWordError::RingOverflow);
        }
        self.ring[self.pos..self.pos + FRAME].copy_from_slice(chunk);
        self.pos = (self.pos + FRAME) % self.ring.len();

        // ---- VAD gate ---------------------------------------------------------
        let speech = self.vad.predict(chunk.iter().copied())? >= self.cfg.vad_threshold;
        if !speech {
            trace!("VAD=0 – skipping frame");
            return Ok(false);
        }
        trace!("VAD=1");

        // ---- feature extraction + score ---------------------------------------
        let feats = features(&self.ring, self.pos);
        let score = self.model.dot(&feats).sigmoid();
        debug!("wake-score = {:.3}", score);

        Ok(score >= self.cfg.wake_threshold)
    }
}
