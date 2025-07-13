//! PCM → log-mel conversion (same parameters as OpenAI Whisper).

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "accelerate",
    feature = "mkl"
))]
use candle::{Result, Tensor};
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "accelerate",
    feature = "mkl"
))]
use candle_transformers::models::whisper::Config;

/// FFT hop length (Whisper constant).
pub const HOP_LENGTH: usize = 160; // 10 ms at 16 kHz
/// Number of frames per 30 s segment.
pub const N_FRAMES: usize = 3000;
/// Original Whisper sample-rate (Hz).
pub const SAMPLE_RATE: usize = 16_000;

/// Convert a slice of PCM samples (mono, f32 –1.0..1.0) to a mel-spectrogram tensor.
pub fn pcm_to_mel(cfg: &Config, pcm: &[f32], mel_filters: &[f32]) -> Vec<f32> {
    use rustfft::{FftPlanner, num_complex::Complex};

    // Hann-windowed STFT.
    let n_fft = cfg.n_fft as usize;
    let n_mels = cfg.num_mel_bins as usize;
    let fft = FftPlanner::<f32>::new().plan_fft_forward(n_fft);

    let mut mel = vec![0f32; N_FRAMES * n_mels];

    for (frame, mel_row) in mel.chunks_mut(n_mels).enumerate() {
        let start = frame * HOP_LENGTH;
        let end = start + n_fft;
        if end > pcm.len() {
            break;
        }

        // Window & FFT.
        let mut buffer: Vec<Complex<f32>> = pcm[start..end]
            .iter()
            .zip(hann_window(n_fft).iter())
            .map(|(&x, w)| Complex { re: x * w, im: 0.0 })
            .collect();
        fft.process(&mut buffer);

        // |FFT|² → Mel.
        for m in 0..n_mels {
            let mut sum = 0f32;
            for (b, &power) in buffer[..n_fft / 2 + 1].iter().enumerate() {
                let w = mel_filters[m * (n_fft / 2 + 1) + b];
                sum += (power.norm_sqr()) * w;
            }
            mel_row[m] = sum.max(1e-10).ln(); // log-energy
        }
    }

    mel
}

/// Pre-computed Hann window cached once.
fn hann_window(n: usize) -> &'static [f32] {
    use std::sync::OnceLock;
    static WIN: OnceLock<Vec<f32>> = OnceLock::new();
    WIN.get_or_init(|| {
        (0..n)
            .map(|i| {
                (std::f32::consts::PI * 2.0 * i as f32 / n as f32)
                    .sin()
                    .powi(2)
            })
            .collect()
    })
}
