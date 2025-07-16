//! PCM → log-mel conversion (same parameters as OpenAI Whisper).

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
    let n_fft = 400; // Whisper uses 400 point FFT (N_FFT constant)
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

/// Build mel filter bank for the given configuration.
///
/// Creates a mel-scale filter bank with triangular filters.
pub fn build_mel_filters(cfg: &Config) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    const N_FFT: usize = 400; // Whisper uses 400 point FFT
    let n_mels = cfg.num_mel_bins;
    let sample_rate = SAMPLE_RATE as f32;
    let fmax = sample_rate / 2.0;

    // Convert frequencies to mel scale
    let mel_min = 0.0;
    let mel_max = hz_to_mel(fmax);

    // Create equally spaced mel frequencies
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert mel frequencies back to Hz
    let freq_points: Vec<f32> = mel_points.iter().map(|&mel| mel_to_hz(mel)).collect();

    // Create filter bank
    let n_freqs = N_FFT / 2 + 1;
    let mut filters = vec![0.0f32; n_mels * n_freqs];

    for m in 0..n_mels {
        let left = freq_points[m];
        let center = freq_points[m + 1];
        let right = freq_points[m + 2];

        for k in 0..n_freqs {
            let freq = k as f32 * sample_rate / N_FFT as f32;

            let weight = if freq >= left && freq <= center {
                if center != left {
                    (freq - left) / (center - left)
                } else {
                    0.0
                }
            } else if freq > center && freq <= right {
                if right != center {
                    (right - freq) / (right - center)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            filters[m * n_freqs + k] = weight;
        }
    }

    Ok(filters)
}

/// Convert frequency in Hz to mel scale
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale value back to Hz
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 2595.0).powf(10.0) - 1.0)
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
