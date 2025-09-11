// src/kc_service.rs
#![cfg_attr(not(feature = "desktop"), allow(dead_code))]

use std::{path::Path, sync::Arc};
use tokio::sync::mpsc;

use crate::{
    KoffeeCandleDetection,
};

/* ─────────────────────────  AUDIO INPUT TRAIT  ─────────────────────── */

use std::future::Future;
use std::pin::Pin;

/// Pulls mono, signed-16-bit PCM @ 16 kHz.
pub trait AudioInput: Send + Sync {
    fn next_chunk(&mut self) -> Pin<Box<dyn Future<Output = Option<Vec<i16>>> + Send + '_>>;
}

#[cfg(feature = "desktop")]
impl AudioInput for CpalMic {
    fn next_chunk(&mut self) -> Pin<Box<dyn Future<Output = Option<Vec<i16>>> + Send + '_>> {
        Box::pin(async move { self.read().await })
    }
}

#[cfg(feature = "desktop")]
type DefaultMic = CpalMic;

/* ──────────────────────────  BUILDER  ─────────────────────────────── */

pub struct KcServiceBuilder<I = DefaultMic> {
    model_path: String,
    mic: Option<I>,
    band_size: u16,
    chan_cap: usize,
    port: u16,
    rx: Option<mpsc::Receiver<KoffeeCandleDetection>>,
}

impl KcServiceBuilder {
    pub fn from_pretrained<P: AsRef<Path>>(path: P) -> Self {
        Self {
            model_path: path.as_ref().to_string_lossy().into_owned(),
            mic: None,
            band_size: 5,
            chan_cap: 16,
            port: 0,
            rx: None,
        }
    }
}

/* ───────────── fluent setters (chain-able) ───────────── */

impl<I: AudioInput + Default> KcServiceBuilder<I> {
    pub fn with_microphone(mut self, mic: I) -> Self {
        self.mic = Some(mic);
        self
    }
    pub fn with_band_size(mut self, n: u16) -> Self {
        self.band_size = n;
        self
    }
    pub fn with_channel_capacity(mut self, n: usize) -> Self {
        self.chan_cap = n;
        self
    }
    pub fn with_port(mut self, p: u16) -> Self {
        self.port = p;
        self
    }
    pub fn with_receiver(mut self, rx: mpsc::Receiver<KoffeeCandleDetection>) -> Self {
        self.rx = Some(rx);
        self
    }

    /// Finishes the pipeline and hands **`KcService`** to the user callback.
    ///
    /// ```rust
    /// # async fn run() -> anyhow::Result<()> {
    /// use koffee_candle::kc_service::KcServiceBuilder;
    /// let (_tx, rx) = tokio::sync::mpsc::channel(32);
    ///
    /// KcServiceBuilder::from_pretrained("assets/hey_rust.kc")
    ///     .with_receiver(rx)
    ///     .listen(|svc| async move { svc.run().await }).await?;
    /// # Ok(()) }
    /// ```
    pub async fn listen<F, Fut>(self, user: F) -> anyhow::Result<()>
    where
        F: FnOnce(KcService<I>) -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<()>>,
    {
        // 1️⃣  detector
        let detector = KcBuilder::new()
            .with_model(&self.model_path)
            .band_size(self.band_size)
            .channel_capacity(self.chan_cap)
            .external_receiver(
                self.rx
                    .ok_or_else(|| anyhow::anyhow!("receiver must be set"))?,
            )
            .build()?;

        // 2️⃣  mic (default if not supplied)
        let mic = match self.mic {
            Some(m) => m,
            None => Default::default(),
        };

        // 3️⃣  service
        let svc = KcService {
            detector: Arc::new(detector),
            mic,
            port: self.port,
        };

        user(svc).await
    }
}

/* ──────────────────────────  SERVICE  ─────────────────────────────── */

pub struct KcService<I = DefaultMic>
where
    I: AudioInput,
{
    detector: Arc<crate::wakewords::DetectorHandle>,
    mic: I,
    port: u16,
}

impl<I: AudioInput> KcService<I> {
    /// Simple async loop – call inside the closure you pass to `listen`.
    pub async fn run(mut self) -> anyhow::Result<()> {
        // Optional: spawn a tiny status HTTP endpoint if `port != 0`.
        if self.port != 0 {
            tokio::spawn(crate::util::mini_status(self.port, self.detector.clone()));
        }

        loop {
            if let Some(pcm) = self.mic.next_chunk().await {
                self.detector.process_i16(&pcm);
            }
        }
    }
}
