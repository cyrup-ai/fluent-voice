//---
// path: src/audio/encoder.rs#![allow(dead_code)]
//---
//! **AudioEncoder**
//!
//! *   Converts arbitrary multi-channel PCM → **mono f32**.
//! *   Optional FFT-based resampling to 16 kHz (or any target SR).
//! *   **Zero-alloc fast-path** when the input sample-rate already matches the target.
//! *   All heap buffers are allocated **once** in [`new`] and re-used.
//!
//! # Highlights
//! * In-place endian conversion + `f32` cast (single `collect`).
//! * `encode_and_resample` works on raw byte slices;
//!   `rencode_and_resample` works on any `&[T: Sample]` without copying.
//! * `reset`/`flush` helpers for stream restarts – no need to rebuild the struct.
//!
//! Unit-tests were moved to **`/tests/audio_encode.rs`** (nextest style).

use crate::audio::{Endianness, Sample, SampleFormat};
use crate::config::AudioFmt;
use rubato::{FftFixedInOut, ResampleError, Resampler, ResamplerConstructionError};

/* ─── public error type ─── */
#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    #[error("unsupported sample-rate")]
    UnsupportedRate,
    #[error(transparent)]
    Resample(#[from] ResampleError),
    #[error(transparent)]
    Construct(#[from] ResamplerConstructionError),
}

type Result<T> = std::result::Result<T, EncoderError>;

/// Re-encode incoming audio to mono `f32` and (optionally) resample.
pub struct AudioEncoder {
    /* ─ immutable source info ─ */
    src_fmt: SampleFormat,
    src_endian: Endianness,
    src_channels: u16,
    bytes_per_input_frame: usize,
    input_samples_per_frame: usize,
    output_samples_per_frame: usize,

    /* ─ reusable rubato state ─ */
    resampler: Option<FftFixedInOut<f32>>,
    resampler_in: Option<Vec<Vec<f32>>>,
    resampler_out: Option<Vec<Vec<f32>>>,
}

impl AudioEncoder {
    /* ─────────── constructors ─────────── */

    /// Create a new [`AudioEncoder`].
    ///
    /// * `fmt`        – input stream format (from [`AudioFmt`])
    /// * `frame_ms`   – frame length in **milliseconds**
    /// * `target_sr`  – desired output sample-rate (e.g. 16 000)
    pub fn new(fmt: &AudioFmt, frame_ms: usize, target_sr: usize) -> Result<Self> {
        let mut in_spf = (fmt.sample_rate * frame_ms / 1_000) * fmt.channels as usize; // samples / frame (input)
        let out_spf = target_sr * frame_ms / 1_000; // samples / frame (out)

        /* optional FFT resampler --------------------------------------------------- */
        let resampler = if fmt.sample_rate != target_sr {
            // `FftFixedInOut::new` now yields a **`ResamplerConstructionError`** which
            // is transparently converted into our `EncoderError::Construct` variant
            // via the `?` operator.
            let rs = FftFixedInOut::<f32>::new(fmt.sample_rate, target_sr, out_spf, 1)?;
            // rubato chooses its own internal frame length; update `in_spf`
            in_spf = rs.input_frames_next() * fmt.channels as usize;
            Some(rs)
        } else {
            None
        };

        let bytes_per_frame = in_spf * fmt.sample_format.get_bytes_per_sample() as usize;

        Ok(Self {
            src_fmt: fmt.sample_format,
            src_endian: fmt.endianness.clone(),
            src_channels: fmt.channels,
            bytes_per_input_frame: bytes_per_frame,
            input_samples_per_frame: in_spf,
            output_samples_per_frame: out_spf,
            resampler_in: resampler.as_ref().map(|r| r.input_buffer_allocate(false)),
            resampler_out: resampler.as_ref().map(|r| r.output_buffer_allocate(true)),
            resampler,
        })
    }

    /* ─────────── public getters ─────────── */

    #[inline]
    pub fn input_bytes(&self) -> usize {
        self.bytes_per_input_frame
    }

    /// *Helper retained for older call-sites that still expect the previous name.*
    #[inline]
    pub fn get_output_frame_length(&self) -> usize {
        self.output_samples_per_frame
    }

    /* ─────────── core API ─────────── */

    /// Encode *raw bytes* → mono `f32` → resample (if needed).
    pub fn encode_and_resample(&mut self, buf: &[u8]) -> Result<Vec<f32>> {
        if buf.len() != self.bytes_per_input_frame {
            return Err(EncoderError::UnsupportedRate);
        }

        let pcm = match self.src_fmt {
            SampleFormat::I8 => decode::<i8>(buf, &self.src_endian),
            SampleFormat::I16 => decode::<i16>(buf, &self.src_endian),
            SampleFormat::I32 => decode::<i32>(buf, &self.src_endian),
            SampleFormat::F32 => decode::<f32>(buf, &self.src_endian),
        };

        self.resample_to_mono(pcm)
    }

    /// Same as [`encode_and_resample`] but works on any `Sample` slice.
    pub fn rencode_and_resample<T: Sample>(&mut self, buf: &[T]) -> Result<Vec<f32>> {
        let pcm: Vec<f32> = buf.iter().copied().map(Sample::into_f32).collect();
        self.resample_to_mono(pcm)
    }

    /* ─────────── streaming helpers ─────────── */

    /// Flush rubato delay-lines – call between unrelated streams.
    #[inline]
    pub fn reset(&mut self) {
        if let Some(rs) = &mut self.resampler {
            rs.reset();
        }
    }

    /* ─────────── internals ─────────── */

    /// Mix-down to mono and pass through the optional resampler.
    fn resample_to_mono(&mut self, pcm: Vec<f32>) -> Result<Vec<f32>> {
        /* channel mix-down (keep first chan) */
        let mono = if self.src_channels == 1 {
            pcm
        } else {
            pcm.chunks_exact(self.src_channels as usize)
                .map(|c| c[0])
                .collect()
        };

        if let Some(rs) = &mut self.resampler {
            // SAFETY: buffers allocated in `new`; length always matches rubato’s contract
            let in_buf = self.resampler_in.as_mut().expect("in-buf not allocated");
            let out_buf = self.resampler_out.as_mut().expect("out-buf not allocated");
            in_buf[0] = mono;
            rs.process_into_buffer(in_buf, out_buf, None)?;
            Ok(out_buf[0].clone())
        } else {
            Ok(mono)
        }
    }
}

/* -------- std::string::String ↔ EncoderError -------------------------- */

impl From<EncoderError> for String {
    #[inline]
    fn from(e: EncoderError) -> Self {
        e.to_string()
    }
}

/* ────────── byte ⇒ sample decoder ────────── */

fn decode<T: Sample>(bytes: &[u8], endian: &Endianness) -> Vec<f32> {
    let sz = T::get_byte_size();
    bytes
        .chunks_exact(sz)
        .map(|chunk| match endian {
            Endianness::Little => T::from_le_bytes(chunk),
            Endianness::Big => T::from_be_bytes(chunk),
            Endianness::Native => T::from_ne_bytes(chunk),
        })
        .map(T::into_f32)
        .collect()
}

/* ─── short helpers used by extractor code ─── */
impl AudioEncoder {
    #[inline]
    pub fn input_samples(&self) -> usize {
        self.input_samples_per_frame
    }
    #[inline]
    pub fn output_samples(&self) -> usize {
        self.output_samples_per_frame
    }
}
