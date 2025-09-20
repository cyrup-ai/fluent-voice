use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use candle_transformers::models::whisper::{self as m};
use flate2::{write::GzEncoder, Compression as GzCompression};
use fluent_voice_domain::VoiceError;
use futures::stream::Stream;
use rand::SeedableRng;
use rand_distr::{weighted::WeightedIndex, Distribution};
use std::io::Write;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokenizers::Tokenizer;

use super::cli::Task;
use super::model::Model;
use fluent_voice_domain::transcription::TranscriptionSegmentImpl;

#[allow(dead_code)] // Used when model loading is re-enabled (line 738 early return currently prevents usage)
#[derive(Debug, Clone)]
pub(super) struct DecodingResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    #[allow(dead_code)]
    pub temperature: f64,
    pub compression_ratio: f64,
}

#[allow(dead_code)] // Used when model loading is re-enabled (line 738 early return currently prevents construction)
pub struct Decoder {
    pub model: Model,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool,
    verbose: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| candle_core::Error::Msg(format!("no token-id for {}", token)))
}

#[allow(dead_code)] // Methods used when model loading is re-enabled (line 738 early return currently prevents usage)
impl Decoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mut model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        verbose: bool,
    ) -> Result<Self> {
        model.reset_kv_cache();
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok())
            .ok_or_else(|| anyhow::anyhow!("unable to find any non-speech token"))?;
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        if self.verbose {
            tracing::debug!(dims = ?audio_features.dims(), "audio features computed");
        }
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = Vec::with_capacity(sample_len);
        tokens.push(self.sot_token);
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;
            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            if i == 0 {
                let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .ok_or_else(|| {
                        VoiceError::ProcessingError("No maximum found in logits".to_string())
                    })?
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle_core::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config().max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        let compression_ratio = if text.is_empty() {
            1.0
        } else {
            let trimmed = text.chars().filter(|c| !c.is_whitespace()).count();
            if trimmed == 0 {
                1.0
            } else {
                let mut buf = Vec::with_capacity(text.len() / 2);
                {
                    let mut encoder = GzEncoder::new(&mut buf, GzCompression::default());
                    encoder.write_all(text.as_bytes()).map_err(E::msg)?;
                    encoder.finish().map_err(E::msg)?;
                }
                (trimmed as f64) / (buf.len() as f64)
            }
        };

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    if self.verbose {
                        tracing::debug!(t = t, error = %err, "Decoding error at temperature");
                    }
                }
            }
        }
        unreachable!()
    }

    pub fn stream_segments<'a>(
        &'a mut self,
        mel: &'a Tensor,
    ) -> impl Stream<Item = Result<TranscriptionSegmentImpl>> + 'a {
        struct SegmentStream<'a> {
            decoder: &'a mut Decoder,
            mel: &'a Tensor,
            seek: usize,
            content_frames: usize,
            pending_chunks: Vec<TranscriptionSegmentImpl>,
        }

        impl<'a> Stream for SegmentStream<'a> {
            type Item = Result<TranscriptionSegmentImpl>;

            fn poll_next(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
            ) -> Poll<Option<Self::Item>> {
                if !self.pending_chunks.is_empty() {
                    return Poll::Ready(Some(Ok(self.pending_chunks.remove(0))));
                }

                if self.seek >= self.content_frames {
                    return Poll::Ready(None);
                }

                let time_offset = (self.seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
                let segment_size = usize::min(self.content_frames - self.seek, m::N_FRAMES);
                let mel_segment = match self.mel.narrow(2, self.seek, segment_size) {
                    Ok(m) => m,
                    Err(e) => return Poll::Ready(Some(Err(e.into()))),
                };
                let segment_duration =
                    (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;

                let dr = match self.decoder.decode_with_fallback(&mel_segment) {
                    Ok(dr) => dr,
                    Err(e) => return Poll::Ready(Some(Err(e))),
                };

                self.seek += segment_size;

                if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD
                    && dr.avg_logprob < m::LOGPROB_THRESHOLD
                {
                    if self.decoder.verbose {
                        tracing::debug!(seek = self.seek, result = ?dr, "no speech detected, skipping");
                    }
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }

                self.pending_chunks.reserve(10);
                if self.decoder.timestamps {
                    let mut tokens_to_decode = Vec::with_capacity(100);
                    let mut prev_timestamp_s = 0.0f64;
                    for &token in dr.tokens.iter() {
                        if token == self.decoder.sot_token || token == self.decoder.eot_token {
                            continue;
                        }
                        if token > self.decoder.no_timestamps_token {
                            let timestamp_s =
                                ((token - self.decoder.no_timestamps_token) as f64) / 50.0;
                            if !tokens_to_decode.is_empty() {
                                match self.decoder.tokenizer.decode(&tokens_to_decode, true) {
                                    Ok(text) => {
                                        self.pending_chunks.push(TranscriptionSegmentImpl::new(
                                            text,
                                            ((time_offset + prev_timestamp_s) * 1000.0) as u32,
                                            ((time_offset + timestamp_s) * 1000.0) as u32,
                                            None, // No speaker ID in this context
                                        ));
                                        tokens_to_decode.clear();
                                    }
                                    Err(e) => return Poll::Ready(Some(Err(E::msg(e)))),
                                }
                            }
                            prev_timestamp_s = timestamp_s;
                        } else {
                            tokens_to_decode.push(token);
                        }
                    }
                    if !tokens_to_decode.is_empty() {
                        if let Ok(text) = self.decoder.tokenizer.decode(&tokens_to_decode, true) {
                            if !text.is_empty() {
                                self.pending_chunks.push(TranscriptionSegmentImpl::new(
                                    text,
                                    ((time_offset + prev_timestamp_s) * 1000.0) as u32,
                                    ((time_offset + segment_duration) * 1000.0) as u32,
                                    None, // No speaker ID in this context
                                ));
                            }
                        }
                    }
                } else {
                    // Create real TranscriptionSegmentImpl with production-quality data from Whisper transcription
                    let text_clone = dr.text.clone(); // Clone before move to avoid borrow after move
                    self.pending_chunks.push(TranscriptionSegmentImpl::new(
                        dr.text,
                        (time_offset * 1000.0) as u32,
                        ((time_offset + segment_duration) * 1000.0) as u32,
                        None, // No speaker ID in this context
                    ));

                    if self.decoder.verbose {
                        tracing::debug!(
                            seek = self.seek,
                            text = %text_clone,
                            avg_logprob = dr.avg_logprob,
                            "Processed segment"
                        );
                    }
                }

                if self.pending_chunks.is_empty() {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                } else {
                    Poll::Ready(Some(Ok(self.pending_chunks.remove(0))))
                }
            }
        }

        SegmentStream {
            decoder: self,
            mel,
            seek: 0,
            content_frames: mel.dims()[2],
            pending_chunks: Vec::with_capacity(10),
        }
    }

    pub fn set_language_token(&mut self, language_token: Option<u32>) {
        self.language_token = language_token;
    }

    /// Detect language using the same algorithm as whisper crate but adapted for our Model enum
    #[allow(dead_code)]
    pub fn detect_language_internal(&mut self, mel: &Tensor) -> candle_core::Result<u32> {
        use candle_core::{IndexOp, D};
        use candle_nn::ops::softmax;
        use candle_transformers::models::whisper as m;

        // Language codes from whisper crate
        const LANGUAGES: [(&str, &str); 99] = [
            ("en", "english"),
            ("zh", "chinese"),
            ("de", "german"),
            ("es", "spanish"),
            ("ru", "russian"),
            ("ko", "korean"),
            ("fr", "french"),
            ("ja", "japanese"),
            ("pt", "portuguese"),
            ("tr", "turkish"),
            ("pl", "polish"),
            ("ca", "catalan"),
            ("nl", "dutch"),
            ("ar", "arabic"),
            ("sv", "swedish"),
            ("it", "italian"),
            ("id", "indonesian"),
            ("hi", "hindi"),
            ("fi", "finnish"),
            ("vi", "vietnamese"),
            ("he", "hebrew"),
            ("uk", "ukrainian"),
            ("el", "greek"),
            ("ms", "malay"),
            ("cs", "czech"),
            ("ro", "romanian"),
            ("da", "danish"),
            ("hu", "hungarian"),
            ("ta", "tamil"),
            ("no", "norwegian"),
            ("th", "thai"),
            ("ur", "urdu"),
            ("hr", "croatian"),
            ("bg", "bulgarian"),
            ("lt", "lithuanian"),
            ("la", "latin"),
            ("mi", "maori"),
            ("ml", "malayalam"),
            ("cy", "welsh"),
            ("sk", "slovak"),
            ("te", "telugu"),
            ("fa", "persian"),
            ("lv", "latvian"),
            ("bn", "bengali"),
            ("sr", "serbian"),
            ("az", "azerbaijani"),
            ("sl", "slovenian"),
            ("kn", "kannada"),
            ("et", "estonian"),
            ("mk", "macedonian"),
            ("br", "breton"),
            ("eu", "basque"),
            ("is", "icelandic"),
            ("hy", "armenian"),
            ("ne", "nepali"),
            ("mn", "mongolian"),
            ("bs", "bosnian"),
            ("kk", "kazakh"),
            ("sq", "albanian"),
            ("sw", "swahili"),
            ("gl", "galician"),
            ("mr", "marathi"),
            ("pa", "punjabi"),
            ("si", "sinhala"),
            ("km", "khmer"),
            ("sn", "shona"),
            ("yo", "yoruba"),
            ("so", "somali"),
            ("af", "afrikaans"),
            ("oc", "occitan"),
            ("ka", "georgian"),
            ("be", "belarusian"),
            ("tg", "tajik"),
            ("sd", "sindhi"),
            ("gu", "gujarati"),
            ("am", "amharic"),
            ("yi", "yiddish"),
            ("lo", "lao"),
            ("uz", "uzbek"),
            ("fo", "faroese"),
            ("ht", "haitian creole"),
            ("ps", "pashto"),
            ("tk", "turkmen"),
            ("nn", "nynorsk"),
            ("mt", "maltese"),
            ("sa", "sanskrit"),
            ("lb", "luxembourgish"),
            ("my", "myanmar"),
            ("bo", "tibetan"),
            ("tl", "tagalog"),
            ("mg", "malagasy"),
            ("as", "assamese"),
            ("tt", "tatar"),
            ("haw", "hawaiian"),
            ("ln", "lingala"),
            ("ha", "hausa"),
            ("ba", "bashkir"),
            ("jw", "javanese"),
            ("su", "sundanese"),
        ];

        let (_bsize, _, seq_len) = mel.dims3()?;
        let mel = mel.narrow(
            2,
            0,
            usize::min(seq_len, self.model.config().max_source_positions),
        )?;
        let device = mel.device();

        // Get language token IDs
        let language_token_ids = LANGUAGES
            .iter()
            .map(|(t, _)| token_id(&self.tokenizer, &format!("<|{t}|>")))
            .collect::<candle_core::Result<Vec<_>>>()?;

        let sot_token = token_id(&self.tokenizer, m::SOT_TOKEN)?;
        let audio_features = self.model.encoder_forward(&mel, true)?;
        let tokens = Tensor::new(&[[sot_token]], device)?;
        let language_token_ids = Tensor::new(language_token_ids.as_slice(), device)?;
        let ys = self.model.decoder_forward(&tokens, &audio_features, true)?;
        let logits = self.model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
        let logits = logits.index_select(&language_token_ids, 0)?;
        let probs = softmax(&logits, D::Minus1)?;
        let probs = probs.to_vec1::<f32>()?;
        let mut probs = LANGUAGES.iter().zip(probs.iter()).collect::<Vec<_>>();
        probs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));

        if self.verbose {
            for ((_, language), p) in probs.iter().take(5) {
                tracing::debug!(language = %language, prob = %p, "language probability");
            }
        }

        let language = token_id(&self.tokenizer, &format!("<|{}|>", probs[0].0 .0))?;
        Ok(language)
    }
}
