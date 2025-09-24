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

#[cfg(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
))]
use rubato::{FftFixedIn, ResampleError, Resampler, ResamplerConstructionError};

// Dummy types for when audio features are disabled
#[cfg(not(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
)))]
#[derive(Debug)]
/// Error type used when audio features are disabled.
///
/// This is a placeholder error type that is used when the crate is compiled
/// without audio processing features enabled.
pub struct ResampleError;

#[cfg(not(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
)))]
impl std::fmt::Display for ResampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Resample error (audio features disabled)")
    }
}

#[cfg(not(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
)))]
impl std::error::Error for ResampleError {}

#[cfg(not(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
)))]
#[derive(Debug)]
/// Error type for resampler construction failures when audio features are disabled.
///
/// This is a placeholder error type that is used when the crate is compiled
/// without audio processing features enabled.
pub struct ResamplerConstructionError;

#[cfg(not(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
)))]
impl std::fmt::Display for ResamplerConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Resampler construction error (audio features disabled)")
    }
}

#[cfg(not(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
)))]
impl std::error::Error for ResamplerConstructionError {}

#[cfg(not(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
)))]
/// Dummy FFT fixed-input resampler used when audio features are disabled.
///
/// This is a placeholder type that provides the same interface as the real
/// FftFixedIn from rubato, but always returns errors since audio processing
/// is disabled.
pub struct FftFixedIn<T>(std::marker::PhantomData<T>);

#[cfg(not(any(
    feature = "microphone",
    feature = "encodec",
    feature = "mimi",
    feature = "snac"
)))]
impl<T> FftFixedIn<T> {
    /// Creates a new FFT fixed-input resampler (disabled version).
    ///
    /// This method always returns an error since audio features are disabled.
    pub fn new(
        _sr_in: usize,
        _sr_out: usize,
        _out_frames: usize,
        _in_channels: usize,
        _out_channels: usize,
    ) -> std::result::Result<Self, ResamplerConstructionError> {
        Err(ResamplerConstructionError)
    }

    /// Resets the resampler state (no-op when audio features are disabled).
    pub fn reset(&mut self) {}

    /// Processes audio through the resampler (disabled version).
    ///
    /// This method always returns an error since audio features are disabled.
    pub fn process_into_buffer(
        &mut self,
        _input: &[Vec<f32>],
        _output: &mut [Vec<f32>],
        _nbr_frames: Option<usize>,
    ) -> std::result::Result<(usize, usize), ResampleError> {
        Err(ResampleError)
    }

    /// Returns the number of input frames needed for the next processing cycle.
    ///
    /// Always returns 0 when audio features are disabled.
    pub fn input_frames_next(&self) -> usize {
        0
    }

    /// Returns the number of output frames that will be produced in the next processing cycle.
    ///
    /// Always returns 0 when audio features are disabled.
    pub fn output_frames_next(&self) -> usize {
        0
    }

    /// Allocates input buffers for audio processing.
    ///
    /// Returns empty buffers when audio features are disabled.
    pub fn input_buffer_allocate(&self, _avoid_reallocation: bool) -> Vec<Vec<f32>> {
        vec![vec![]]
    }

    /// Allocates output buffers for audio processing.
    ///
    /// Returns empty buffers when audio features are disabled.
    pub fn output_buffer_allocate(&self, _avoid_reallocation: bool) -> Vec<Vec<f32>> {
        vec![vec![]]
    }
}

/* ─── public error type ─── */
/// Errors that can occur during audio encoding operations.
#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    /// Sample rate is not supported by the encoder.
    #[error("unsupported sample-rate")]
    UnsupportedRate,
    /// Error during resampling operation.
    #[error(transparent)]
    Resample(#[from] ResampleError),
    /// Error during resampler construction.
    #[error(transparent)]
    Construct(#[from] ResamplerConstructionError),
    /// Generic error with custom message.
    #[error("{0}")]
    Generic(String),
}

impl From<&str> for EncoderError {
    fn from(msg: &str) -> Self {
        EncoderError::Generic(msg.to_string())
    }
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
    resampler: Option<FftFixedIn<f32>>,
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
        #[cfg(any(
            feature = "microphone",
            feature = "encodec",
            feature = "mimi",
            feature = "snac"
        ))]
        let mut in_spf = (fmt.sample_rate * frame_ms / 1_000) * fmt.channels as usize; // samples / frame (input)
        #[cfg(not(any(
            feature = "microphone",
            feature = "encodec",
            feature = "mimi",
            feature = "snac"
        )))]
        let in_spf = (fmt.sample_rate * frame_ms / 1_000) * fmt.channels as usize; // samples / frame (input)
        let out_spf = target_sr * frame_ms / 1_000; // samples / frame (out)

        /* optional FFT resampler --------------------------------------------------- */
        #[cfg(any(
            feature = "microphone",
            feature = "encodec",
            feature = "mimi",
            feature = "snac"
        ))]
        let resampler = if fmt.sample_rate != target_sr {
            // `FftFixedIn::new` now yields a **`ResamplerConstructionError`** which
            // is transparently converted into our `EncoderError::Construct` variant
            // via the `?` operator.
            // Use input format channels to match the actual audio format
            let rs = FftFixedIn::<f32>::new(
                fmt.sample_rate,
                target_sr,
                out_spf,
                fmt.channels as usize,
                fmt.channels as usize,
            )?;
            // rubato chooses its own internal frame length; update `in_spf`
            in_spf = rs.input_frames_next() * fmt.channels as usize;
            Some(rs)
        } else {
            None
        };

        #[cfg(not(any(
            feature = "microphone",
            feature = "encodec",
            feature = "mimi",
            feature = "snac"
        )))]
        let resampler: Option<FftFixedIn<f32>> = None;

        let bytes_per_frame = in_spf * fmt.sample_format.get_bytes_per_sample() as usize;

        // Ensure proper buffer allocation based on actual channel count
        #[cfg(any(
            feature = "microphone",
            feature = "encodec",
            feature = "mimi",
            feature = "snac"
        ))]
        let resampler_in = resampler.as_ref().map(|r| {
            let mut buffers = r.input_buffer_allocate(false);
            // Ensure we have the correct number of channel buffers
            let expected_channels = fmt.channels as usize;
            if buffers.len() != expected_channels {
                buffers.resize(expected_channels, Vec::with_capacity(r.input_frames_next()));
            }
            // Pre-allocate all channel buffers to avoid size 0 issues
            for buffer in &mut buffers {
                if buffer.capacity() == 0 {
                    buffer.reserve(r.input_frames_next());
                }
            }
            buffers
        });

        #[cfg(not(any(
            feature = "microphone",
            feature = "encodec",
            feature = "mimi",
            feature = "snac"
        )))]
        let resampler_in = None;

        #[cfg(any(
            feature = "microphone",
            feature = "encodec",
            feature = "mimi",
            feature = "snac"
        ))]
        let resampler_out = resampler.as_ref().map(|r| {
            let mut buffers = r.output_buffer_allocate(true);
            // Output is always mono (1 channel) after mix-down
            if buffers.len() != 1 {
                buffers.resize(1, Vec::with_capacity(r.output_frames_next()));
            }
            // Pre-allocate the buffer to avoid size 0 issues
            if buffers[0].capacity() == 0 {
                buffers[0].reserve(r.output_frames_next());
            }
            buffers
        });

        #[cfg(not(any(
            feature = "microphone",
            feature = "encodec",
            feature = "mimi",
            feature = "snac"
        )))]
        let resampler_out = None;

        Ok(Self {
            src_fmt: fmt.sample_format,
            src_endian: fmt.endianness.clone(),
            src_channels: fmt.channels,
            bytes_per_input_frame: bytes_per_frame,
            input_samples_per_frame: in_spf,
            output_samples_per_frame: out_spf,
            resampler_in,
            resampler_out,
            resampler,
        })
    }

    /* ─────────── public getters ─────────── */

    /// Returns the number of bytes per input frame.
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
            // SAFETY: buffers allocated in `new`; length always matches rubato's contract
            let in_buf = match self.resampler_in.as_mut() {
                Some(buf) => buf,
                None => return Err("Resampler input buffer not allocated".into()),
            };
            let out_buf = match self.resampler_out.as_mut() {
                Some(buf) => buf,
                None => return Err("Resampler output buffer not allocated".into()),
            };

            // Handle channel assignment based on actual channel count
            if self.src_channels == 1 {
                // Mono input: assign to first channel
                in_buf[0] = mono;
            } else {
                // Multi-channel input: distribute mono data across all expected channels
                for i in 0..self.src_channels as usize {
                    if i < in_buf.len() {
                        in_buf[i] = mono.clone();
                    }
                }
            }

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
    /// Returns the number of input samples per frame.
    #[inline]
    pub fn input_samples(&self) -> usize {
        self.input_samples_per_frame
    }
    /// Returns the number of output samples per frame.
    #[inline]
    pub fn output_samples(&self) -> usize {
        self.output_samples_per_frame
    }
}
