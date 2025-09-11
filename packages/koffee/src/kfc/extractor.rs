//---
// path: potter-dsp/src/kfc.rs
//---
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::many_single_char_names)]

use rustfft::{FftPlanner, num_complex::Complex32};
use std::f32::consts::PI;

/// Returned whenever the `Kfc` API is mis-used (wrong frame size, etc.).
#[derive(Debug, thiserror::Error)]
pub enum KfcError {
    /// Frame length does not match the configured frame size.
    #[error("frame length ({given}) must equal configured frame size ({expected})")]
    BadFrame {
        /// The provided frame length.
        given: usize,
        /// The expected frame length.
        expected: usize,
    },
    /// Output slice is too small for the extracted features.
    #[error("output slice too small (need {need}, got {got})")]
    OutTooSmall {
        /// Required output slice size.
        need: usize,
        /// Actual output slice size.
        got: usize,
    },
}

/// Streaming **Mel-frequency cepstral coefficient** extractor.
///
/// *No* heap allocation occurs while extracting –
/// all scratch buffers are pre-allocated in `new`.
///
/// ```no_run
/// # use potter_dsp::kfc::Kfc;
/// # const N: usize = 480;              // 30 ms @ 16 kHz
/// let mut kfc = Kfc::<13>::new(16_000, N, 10, 40)?; // 13 coeffs
/// let frame     = [0f32; N];          // fill from your audio callback
/// let mut out   = [0f32; 13];
/// kfc.extract(&frame, &mut out)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Kfc<const COEFFS: usize> {
    // configuration
    #[allow(dead_code)]
    sample_rate: usize,
    frame_size: usize, // PCM samples per frame
    shift_size: usize,

    // cached DSP bits
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    fft_buf: Vec<Complex32>,
    hamming: Vec<f32>,
    filter_bank: Vec<Vec<f32>>, // [mel_bin][mag_bin]

    // scratch
    mag_spectrum: Vec<f32>, // reused between calls
}

impl<const COEFFS: usize> Kfc<COEFFS> {
    /// Create a new extractor.
    ///
    /// * `frame_size` – samples per analysis frame (e.g. 480 = 30 ms @ 16 kHz)
    /// * `shift_ms`   – hop size in **milliseconds** (commonly 10 ms)
    /// * `mel_bins`   – number of triangular mel filters (≈ 40)
    pub fn new(
        sample_rate: usize,
        frame_size: usize,
        shift_ms: usize,
        mel_bins: usize,
    ) -> Result<Self, KfcError> {
        let shift_size = sample_rate * shift_ms / 1_000; // samples
        // --- pre-compute ---
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(frame_size);
        let hamming = (0..frame_size)
            .map(|n| 0.54 - 0.46 * ((2.0 * PI * n as f32) / (frame_size - 1) as f32).cos())
            .collect::<Vec<_>>();
        let mag_bins = frame_size / 2;
        let filter_bank = mel_filter_bank(sample_rate, mag_bins, mel_bins);
        Ok(Self {
            sample_rate,
            frame_size,
            shift_size,
            fft,
            fft_buf: vec![Complex32::ZERO; frame_size],
            hamming,
            filter_bank,
            mag_spectrum: vec![0.0; mag_bins],
        })
    }

    /// Size (in **samples**) expected by [`extract`].
    #[inline]
    #[allow(dead_code)]
    pub const fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Hop length (in samples) that corresponds to the `shift_ms` given at construction.
    #[inline]
    #[allow(dead_code)]
    pub const fn shift_size(&self) -> usize {
        self.shift_size
    }

    /// Compute **COEFFS** KFCs for a single `frame` of *mono 32-bit float* PCM.
    ///
    /// The first (0-th) cepstral coefficient is **dropped** (energy) – so
    /// `out.len()` must equal `COEFFS`.
    pub fn extract(&mut self, frame: &[f32], out: &mut [f32]) -> Result<(), KfcError> {
        if frame.len() != self.frame_size {
            return Err(KfcError::BadFrame {
                given: frame.len(),
                expected: self.frame_size,
            });
        }
        if out.len() < COEFFS {
            return Err(KfcError::OutTooSmall {
                need: COEFFS,
                got: out.len(),
            });
        }
        // 1) Window + FFT
        for (dst, (&x, &w)) in self.fft_buf.iter_mut().zip(frame.iter().zip(&self.hamming)) {
            dst.re = x * w;
            dst.im = 0.0;
        }
        self.fft.process(&mut self.fft_buf);

        // 2) |FFT| -> magnitude spectrum
        let mags = &mut self.mag_spectrum;
        for (i, m) in mags.iter_mut().enumerate() {
            let c = self.fft_buf[i];
            *m = (c.re * c.re + c.im * c.im).sqrt();
        }

        // 3) Apply mel filter bank  -> log energies
        let mut mel_energies = vec![0f32; COEFFS + 1]; // +1 because we drop c0 later
        for (mel_bin, filt) in self.filter_bank.iter().enumerate().take(COEFFS + 1) {
            let e = filt
                .iter()
                .zip(mags.iter())
                .map(|(f, &m)| f * m)
                .sum::<f32>()
                + f32::MIN_POSITIVE;
            mel_energies[mel_bin] = e.ln();
        }

        // 4) DCT-II.  We skip coefficient 0 (energy) and keep the rest.
        let n = (COEFFS + 1) as f32;
        for k in 1..=COEFFS {
            let mut s = 0.0;
            for (m, &e) in mel_energies.iter().enumerate() {
                s += e * ((PI / n) * (m as f32 + 0.5) * k as f32).cos();
            }
            out[k - 1] = 2.0 * s;
        }
        Ok(())
    }
}

// ---------- helpers --------------------------------------------------------

fn mel_filter_bank(sr: usize, mag_bins: usize, mel_bins: usize) -> Vec<Vec<f32>> {
    let f_max = sr as f32 / 2.0;
    let mel_max = freq_to_mel(f_max);
    let mel_step = mel_max / (mel_bins + 1) as f32;
    let mut bank = vec![vec![0f32; mag_bins]; mel_bins];

    // mel triangular windows
    let center_freqs: Vec<f32> = (0..=mel_bins + 1)
        .map(|i| mel_to_freq(i as f32 * mel_step))
        .collect();

    for (i, filt) in bank.iter_mut().enumerate() {
        let f_left = center_freqs[i];
        let f_center = center_freqs[i + 1];
        let f_right = center_freqs[i + 2];

        for (bin, amp) in filt.iter_mut().enumerate() {
            let freq = bin as f32 * f_max / (mag_bins - 1) as f32;
            *amp = if freq < f_left || freq > f_right {
                0.0
            } else if freq <= f_center {
                (freq - f_left) / (f_center - f_left)
            } else {
                (f_right - freq) / (f_right - f_center)
            };
        }
    }
    bank
}

#[inline]
fn freq_to_mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}
#[inline]
fn mel_to_freq(m: f32) -> f32 {
    700.0 * ((m / 1127.0).exp() - 1.0)
}

/// The default extractor instance, fixed at 13 coefficients
pub type KfcExtractor = Kfc<13>;

// Production implementation of the compute method for wav_file_extractor
impl<const COEFFS: usize> Kfc<COEFFS> {
    /// Process audio chunk and extract KFC frames.
    /// Returns an iterator of KFC frames (vectors of coefficients).
    ///
    /// This method processes a full audio chunk and extracts KFCs, handling
    /// the chunking at the frame_size level and returning an iterator of
    /// coefficient vectors.
    pub fn compute(&mut self, chunk: &[f32]) -> impl Iterator<Item = Vec<f32>> + '_ {
        // Calculate how many full frames we can extract from this chunk
        let num_frames = if chunk.len() >= self.frame_size {
            // At least one full frame
            1 + (chunk.len() - self.frame_size) / self.shift_size
        } else {
            // Not enough data for a single frame
            0
        };

        // Pre-allocate frames for all frames we'll extract
        let mut frames = Vec::with_capacity(num_frames);

        // Process each frame with proper shifting
        for i in 0..num_frames {
            let start = i * self.shift_size;
            let end = start + self.frame_size;

            // Ensure we don't go past the end of the chunk
            if end <= chunk.len() {
                let mut kfc_frame = vec![0.0; COEFFS];

                // Extract KFCs from this frame
                if self.extract(&chunk[start..end], &mut kfc_frame).is_ok() {
                    frames.push(kfc_frame);
                }
            }
        }

        // Return iterator of all extracted frames
        frames.into_iter()
    }
}
