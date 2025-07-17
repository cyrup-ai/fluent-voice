//---
// path: src/kfc/wav_extractor.rs
//---
//! **KFC feature extraction from WAV files**
//!
//! 1. Stream-decode WAV → mono `f32` at 16 kHz (via [`AudioEncoder`]).
//! 2. Slice into *overlapping* frames (`FRAME_LEN_MS` / `FRAME_SHIFT_MS`).
//! 3. Run the [`KfcExtractor`] front-end → **KFC coefficients**.
//! 4. Mean-centre & return the frame matrix **plus median RMS level**.
//!
//! All heap buffers are allocated once and re-used; there are **no `unwrap`s**.

use std::io::BufReader;

use hound::{Sample as HoundSample, WavReader, WavSpec};

use thiserror::Error;

use crate::{
    audio::{AudioEncoder, Endianness, GainNormalizerFilter, Sample, SampleFormat},
    config::AudioFmt,
    constants::{
        DETECTOR_INTERNAL_SAMPLE_RATE, KFCS_EXTRACTOR_FRAME_LENGTH_MS,
        KFCS_EXTRACTOR_FRAME_SHIFT_MS,
    },
};

use super::KfcNormalizer;
use super::extractor::{KfcError, KfcExtractor};

/* ─────────────────────── error handling ─────────────────────── */

#[derive(Debug, Error)]
pub enum ExtractorError {
    #[error("wav: {0}")]
    Wav(#[from] hound::Error),
    #[error("audio encode: {0}")]
    Encode(String),
    #[error("unsupported wav format")]
    Unsupported,
    #[error("kfc: {0}")]
    Kfc(#[from] KfcError),
}

impl From<crate::audio::EncoderError> for ExtractorError {
    fn from(e: crate::audio::EncoderError) -> Self {
        Self::Encode(e.to_string())
    }
}

/* ─────────────────────── public API ─────────────────────────── */

pub(crate) struct KfcWavFileExtractor;

impl KfcWavFileExtractor {
    /// Extract **mean-centred KFC frames** from a WAV reader.
    ///
    /// * `out_rms` – median RMS level of all frames is written back here.
    pub(crate) fn compute_kfcs<R: std::io::Read>(
        reader: BufReader<R>,
        out_rms: &mut f32,
        _kfc_size: u16, // kept for signature parity (unused internally)
    ) -> Result<Vec<Vec<f32>>, ExtractorError> {
        /* ---------- 1. Open WAV & build encoder ------------------------- */
        let mut wav = WavReader::new(reader)?;
        let fmt: AudioFmt = wav.spec().try_into()?;
        let mut encoder = AudioEncoder::new(
            &fmt,
            KFCS_EXTRACTOR_FRAME_LENGTH_MS,
            DETECTOR_INTERNAL_SAMPLE_RATE,
        )?;

        /* ---------- 2. Build KFC front-end ------------------------------ */
        let samples_per_frame = encoder.output_samples();
        let mut kfc = KfcExtractor::new(
            DETECTOR_INTERNAL_SAMPLE_RATE,
            samples_per_frame,
            KFCS_EXTRACTOR_FRAME_SHIFT_MS, // **milliseconds**
            40,                            // mel bins
        )?;

        let mut rms_levels = smallvec::SmallVec::<f32, 128>::new();
        let mut frames: Vec<Vec<f32>> = Vec::new();

        /* ---------- 3. Stream-decode & process -------------------------- */
        match fmt.sample_format {
            SampleFormat::I8 => Self::read_samples::<_, i8>(
                &mut wav,
                &mut encoder,
                &mut kfc,
                &mut rms_levels,
                &mut frames,
            ),
            SampleFormat::I16 => Self::read_samples::<_, i16>(
                &mut wav,
                &mut encoder,
                &mut kfc,
                &mut rms_levels,
                &mut frames,
            ),
            SampleFormat::I32 => Self::read_samples::<_, i32>(
                &mut wav,
                &mut encoder,
                &mut kfc,
                &mut rms_levels,
                &mut frames,
            ),
            SampleFormat::F32 => Self::read_samples::<_, f32>(
                &mut wav,
                &mut encoder,
                &mut kfc,
                &mut rms_levels,
                &mut frames,
            ),
        }?;

        /* ---------- 4. Post-process ------------------------------------ */
        if !rms_levels.is_empty() {
            *out_rms = median(&mut rms_levels);
        }
        KfcNormalizer::normalize(&mut frames);
        Ok(frames)
    }

    /* -------- helpers ------------------------------------------------- */

    fn read_samples<R, S>(
        wav: &mut WavReader<R>,
        encoder: &mut AudioEncoder,
        kfc: &mut KfcExtractor,
        rms_levels: &mut smallvec::SmallVec<f32, 128>,
        out_frames: &mut Vec<Vec<f32>>,
    ) -> Result<(), ExtractorError>
    where
        R: std::io::Read,
        S: HoundSample + Sample,
    {
        // Ensure we have sufficient buffer capacity for input samples
        let required_capacity = encoder.input_samples().max(480);
        let mut in_buf = Vec::<S>::with_capacity(required_capacity);

        // Pre-allocate to avoid buffer size issues during processing
        if in_buf.capacity() == 0 {
            in_buf.reserve(required_capacity);
        }

        // Buffer to accumulate resampled audio until we have enough for KFC (480 samples)
        let kfc_frame_size = 480; // 30ms at 16kHz
        let mut kfc_buffer = Vec::<f32>::new();

        for sample in wav.samples::<S>() {
            let s = sample?;
            in_buf.push(s);

            if in_buf.len() == encoder.input_samples() {
                /* bytes → mono f32 (resampled) */
                let samples = encoder.rencode_and_resample(&in_buf)?;
                in_buf.clear();

                rms_levels.push(GainNormalizerFilter::get_rms_level(&samples));

                // Add resampled samples to KFC buffer
                kfc_buffer.extend_from_slice(&samples);

                // Process complete KFC frames when we have enough samples
                while kfc_buffer.len() >= kfc_frame_size {
                    let frame: Vec<f32> = kfc_buffer.drain(..kfc_frame_size).collect();
                    for v in kfc.compute(&frame) {
                        out_frames.push(v);
                    }
                }
            }
        }
        Ok(())
    }
}

/* ────────────────────── utils ──────────────────────────────── */

fn median(buf: &mut [f32]) -> f32 {
    buf.sort_by(|a, b| a.total_cmp(b));
    buf[buf.len() / 2]
}

/* ---------- WAV spec → AudioFmt conversion ------------------- */

impl TryFrom<WavSpec> for AudioFmt {
    type Error = ExtractorError;

    fn try_from(spec: WavSpec) -> Result<Self, Self::Error> {
        let sample_format = match spec.sample_format {
            hound::SampleFormat::Int => SampleFormat::int_of_size(spec.bits_per_sample),
            hound::SampleFormat::Float => SampleFormat::float_of_size(spec.bits_per_sample),
        }
        .ok_or(ExtractorError::Unsupported)?;

        Ok(AudioFmt {
            channels: spec.channels,
            sample_format,
            sample_rate: spec.sample_rate as usize,
            endianness: Endianness::Little,
        })
    }
}
