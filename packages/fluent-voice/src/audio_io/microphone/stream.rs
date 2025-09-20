use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_transformers::models::whisper::{self as m, Config};

// Enable real resampler imports (using FastFixedIn like whisper package)
#[cfg(feature = "microphone")]
use rubato::{FastFixedIn, PolynomialDegree, Resampler};

// Import whisper audio processing
use candle_transformers::models::whisper::audio::pcm_to_mel;

use futures::stream::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokenizers::Tokenizer;

use super::decoder::{token_id, Decoder};
use fluent_voice_domain::transcription::TranscriptionSegmentImpl;

#[allow(dead_code)] // Used when model loading is re-enabled
pub struct AudioStream {
    pub decoder: Decoder,
    pub tokenizer: Tokenizer,
    pub config: Config,
    pub mel_filters: Vec<f32>,
    pub rx: std::sync::mpsc::Receiver<Vec<f32>>,
    #[cfg(feature = "microphone")]
    pub resampler: Option<FastFixedIn<f32>>,
    #[cfg(not(feature = "microphone"))]
    pub resampler: Option<()>, // Placeholder for non-microphone builds
    pub in_sample_rate: usize,
    pub buffered_pcm: Vec<f32>,
    pub language_token_set: bool,
    pub device: Device,
    pub verbose: bool,
    pub model_is_multilingual: bool,
    pub language: Option<String>,
}

#[allow(dead_code)] // Used when model loading is re-enabled
impl AudioStream {
    pub fn new(
        decoder: Decoder,
        tokenizer: Tokenizer,
        config: Config,
        mel_filters: Vec<f32>,
        rx: std::sync::mpsc::Receiver<Vec<f32>>,
        in_sample_rate: usize,
        device: Device,
        verbose: bool,
        model_is_multilingual: bool,
        language: Option<String>,
    ) -> Result<Self> {
        // Create real resampler if needed (using proven whisper package pattern)
        let resampler = if in_sample_rate != m::SAMPLE_RATE {
            #[cfg(feature = "microphone")]
            {
                let resample_ratio = m::SAMPLE_RATE as f64 / in_sample_rate as f64;
                Some(FastFixedIn::<f32>::new(
                    resample_ratio,           // ratio (e.g., 16000.0 / 44100.0)
                    10.,                      // max_resample_ratio_relative
                    PolynomialDegree::Septic, // quality setting
                    1024,                     // chunk_size
                    1,                        // mono channel
                )?)
            }
            #[cfg(not(feature = "microphone"))]
            None
        } else {
            None
        };

        Ok(Self {
            decoder,
            tokenizer,
            config,
            mel_filters,
            rx,
            resampler,
            in_sample_rate,
            buffered_pcm: Vec::with_capacity(30 * in_sample_rate),
            language_token_set: false,
            device,
            verbose,
            model_is_multilingual,
            language,
        })
    }
}
#[allow(dead_code)] // Used when model loading is re-enabled
impl Stream for AudioStream {
    type Item = Result<TranscriptionSegmentImpl>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let chunk = match self.rx.try_recv() {
            Ok(chunk) => chunk,
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
            Err(_) => return Poll::Ready(None),
        };

        self.buffered_pcm.extend_from_slice(&chunk);
        if self.buffered_pcm.len() < 10 * self.in_sample_rate {
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }

        // Extract immutable data before mutable borrow to avoid borrow checker conflicts
        let in_sample_rate = self.in_sample_rate;
        let buffered_pcm_len = self.buffered_pcm.len();

        let pcm = if self.resampler.is_some() {
            self.process_with_resampler(in_sample_rate, buffered_pcm_len)?
        } else {
            std::mem::take(&mut self.buffered_pcm)
        };

        self.process_audio_chunk(pcm, cx)
    }
}
#[allow(dead_code)] // Used when model loading is re-enabled
impl AudioStream {
    fn process_with_resampler(
        &mut self,
        _in_sample_rate: usize,
        buffered_pcm_len: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "microphone")]
        {
            if let Some(ref mut resampler) = self.resampler {
                // Use exact pattern from whisper package (lines 460-480)
                let chunk_size = 1024; // Fixed chunk size as used in whisper
                let full_chunks = buffered_pcm_len / chunk_size;
                let remainder = buffered_pcm_len % chunk_size;
                let mut resampled_pcm = Vec::new();

                // Process full chunks using proven whisper pattern
                for chunk in 0..full_chunks {
                    let buffered_pcm = &self.buffered_pcm[chunk * 1024..(chunk + 1) * 1024];
                    let pcm = resampler.process(&[&buffered_pcm], None)?;
                    resampled_pcm.extend_from_slice(&pcm[0]);
                }

                // Handle remaining PCM data (whisper pattern)
                if remainder == 0 {
                    self.buffered_pcm.clear();
                } else {
                    // efficiently copy the remainder to the beginning and truncate
                    self.buffered_pcm.copy_within(full_chunks * chunk_size.., 0);
                    self.buffered_pcm.truncate(remainder);
                }

                Ok(resampled_pcm)
            } else {
                // No resampler needed - return original data
                Ok(std::mem::take(&mut self.buffered_pcm))
            }
        }
        #[cfg(not(feature = "microphone"))]
        {
            // Fallback for non-microphone builds
            Ok(std::mem::take(&mut self.buffered_pcm))
        }
    }

    fn process_audio_chunk(
        &mut self,
        pcm: Vec<f32>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<TranscriptionSegmentImpl>>> {
        // REAL mel spectrogram computation using candle-transformers

        // Ensure PCM is the right length for whisper (30 seconds = 480,000 samples at 16kHz)
        let target_length = m::N_SAMPLES; // 480,000 samples
        let mut processed_pcm = pcm;

        // Pad or truncate to target length
        processed_pcm.resize(target_length, 0.0);

        // Generate real mel spectrogram using candle-transformers
        let mel_vec = pcm_to_mel(&self.config, &processed_pcm, &self.mel_filters);
        let mel_len = mel_vec.len();
        let frames = mel_len / self.config.num_mel_bins;

        let mel =
            match Tensor::from_vec(mel_vec, (1, self.config.num_mel_bins, frames), &self.device) {
                Ok(m) => m,
                Err(e) => return Poll::Ready(Some(Err(e.into()))),
            };

        if !self.language_token_set {
            match self.set_language_token() {
                Ok(()) => {}
                Err(e) => return Poll::Ready(Some(Err(e))),
            }
        }

        let mut segment_stream = self.decoder.stream_segments(&mel);
        let poll_result = Pin::new(&mut segment_stream).poll_next(cx);
        // Drop segment_stream before accessing self.decoder.model to avoid borrow conflicts
        drop(segment_stream);

        match poll_result {
            Poll::Ready(Some(Ok(chunk))) => {
                self.decoder.model.reset_kv_cache();
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                self.decoder.model.reset_kv_cache();
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Poll::Pending => Poll::Pending,
        }
    }
    fn set_language_token(&mut self) -> Result<()> {
        let language_token = match (self.model_is_multilingual, self.language.clone()) {
            (true, None) => {
                // Use token_id for English as fallback when language detection isn't available
                let language_result = token_id(&self.tokenizer, "<|en|>");
                match language_result {
                    Ok(result) => {
                        if self.verbose {
                            tracing::debug!(result = ?result, "Detected language");
                        }
                        Some(result) // Use token directly
                    }
                    Err(e) => return Err(anyhow::Error::from(e)),
                }
            }
            (true, Some(lang)) => match token_id(&self.tokenizer, &format!("<|{}|>", lang)) {
                Ok(token) => Some(token),
                Err(e) => return Err(e.into()),
            },
            (false, None) => None,
            (false, Some(_)) => {
                return Err(anyhow::anyhow!(
                    "language cannot be set for non-multilingual model"
                ));
            }
        };
        self.decoder.set_language_token(language_token);
        self.language_token_set = true;
        Ok(())
    }
}
