//! Fluent, ergonomic interface for launching a Whisper transcription.
//!
//! # Example
//! ```no_run
//! use fluent_voice::prelude::*;
//! use whisper::{Whisper, WhisperStream};
//!
//! // STT trait API
//! let stream = Whisper::builder()
//!     .with_source(SpeechSource::File {
//!         path: "./assets/audio.mp3".into(),
//!         format: AudioFormat::Mp3Khz44_128,
//!     })
//!     .vad_mode(VadMode::Accurate)
//!     .listen(|conversation| {
//!         Ok(conv) => conv.into_stream(),
//!         Err(e) => Err(e),
//!     })
//!     .await?;
//!
//! // Legacy convenience API (still works)
//! let text = Whisper::transcribe("./assets/audio.mp3")
//!     .with_progress("{file} :: {percent}%")
//!     .emit()                               // => WhisperStream
//!     .collect_with(|result| match result { // => String
//!         Ok(t)  => t.as_text(),
//!         Err(e) => format!("transcription failed: {e}"),
//!     })
//!     .await;
//! ```
//!
//! Internally the heavy lifting is off-loaded to a `spawn_blocking`
//! worker so that the public `emit()` remains **synchronous** and lazy
//! — no CPU work starts until the returned stream is first polled.

#![cfg(feature = "tokio")]

use core::future::Future;
use futures::{Stream, StreamExt};
use tokio::sync::mpsc;

use crate::{pcm_decode, stream::WhisperStream, transcript::Transcript, types::TtsChunk};

use fluent_voice::{
    audio_format::AudioFormat,
    language::Language,
    mic_backend::MicBackend,
    noise_reduction::NoiseReduction,
    speech_source::SpeechSource,
    stt_conversation::{SttConversation, SttConversationBuilder, SttConversationExt},
    stt_engine::SttEngine,
    timestamps::{Diarization, Punctuation, TimestampsGranularity, WordTimestamps},
    vad_mode::VadMode,
    voice_error::VoiceError,
};

/// Zero-sized helper used purely for its associated `transcribe()`
/// function. Mirrors the familiar `std::fs::File::open` style API.
pub struct Whisper;

impl Whisper {
    /// Begin building a transcription job from a file path (convenience API).
    pub fn transcribe<P: Into<String>>(path: P) -> WhisperBuilder {
        WhisperBuilder::new().with_source(SpeechSource::File {
            path: path.into(),
            format: AudioFormat::Mp3Khz44_128, // Default, could be auto-detected
        })
    }
}

/// Builder for Whisper transcription with both STT trait and convenience APIs.
#[derive(Debug, Clone)]
pub struct WhisperBuilder {
    // Core source
    source: Option<SpeechSource>,

    // STT configuration
    vad_mode: VadMode,
    noise_reduction: NoiseReduction,
    language_hint: Option<Language>,
    diarization: Diarization,
    word_timestamps: WordTimestamps,
    timestamps_granularity: TimestampsGranularity,
    punctuation: Punctuation,

    // Legacy convenience options
    progress_template: Option<String>,
}

impl WhisperBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            source: None,
            vad_mode: VadMode::Accurate,
            noise_reduction: NoiseReduction::Low,
            language_hint: None,
            diarization: Diarization::Off,
            word_timestamps: WordTimestamps::Off,
            timestamps_granularity: TimestampsGranularity::Word,
            punctuation: Punctuation::On,
            progress_template: None,
        }
    }

    /// Attach a progress message template (convenience API).
    /// Use `{file}` and `{percent}` placeholders.
    pub fn with_progress<S: Into<String>>(mut self, template: S) -> Self {
        self.progress_template = Some(template.into());
        self
    }

    /// Produce a *lazy* `WhisperStream` (convenience API).
    ///
    /// No decoding occurs until the caller begins to poll the stream.
    pub fn emit(self) -> WhisperStream {
        let (tx, rx) = mpsc::unbounded_channel::<TtsChunk>();

        let source = self.source.expect("No source specified");
        let tmpl = self.progress_template.clone();

        tokio::spawn(async move {
            // Off-load CPU to a dedicated blocking thread.
            if let Err(e) = tokio::task::spawn_blocking(move || {
                internal::decode_to_stream(&source, tmpl.as_deref(), tx)
            })
            .await
            {
                eprintln!("whisper worker panicked: {e}");
            }
        });

        WhisperStream::new(rx)
    }

    /// Drain the stream and gather a [`Transcript`].
    pub async fn collect(self) -> anyhow::Result<Transcript> {
        let mut transcript = Transcript::default();
        let mut stream = self.emit_stream();
        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => transcript.push(chunk),
                Err(e) => return Err(anyhow::anyhow!("Transcription error: {}", e)),
            }
        }
        Ok(transcript)
    }

    /// Variant that accepts a user-supplied closure to post-process the
    /// result (success or failure) in one go.
    pub async fn collect_with<F, R>(self, handler: F) -> R
    where
        F: FnOnce(anyhow::Result<Transcript>) -> R,
    {
        let res = self.collect().await;
        handler(res)
    }

    /// Convenience: immediately obtain a `Stream<Item = String>` with
    /// only the plain text of each chunk.
    pub fn as_text(self) -> impl Stream<Item = String> {
        self.emit_stream().filter_map(|chunk_result| async move {
            match chunk_result {
                Ok(chunk) => Some(chunk.text),
                Err(_) => None,
            }
        })
    }
}

impl SttConversationBuilder for WhisperBuilder {
    type Conversation = WhisperConversation;
    type Transcript = Transcript;

    fn with_source(mut self, src: SpeechSource) -> Self {
        self.source = Some(src);
        self
    }

    fn vad_mode(mut self, mode: VadMode) -> Self {
        self.vad_mode = mode;
        self
    }

    fn noise_reduction(mut self, level: NoiseReduction) -> Self {
        self.noise_reduction = level;
        self
    }

    fn language_hint(mut self, lang: Language) -> Self {
        self.language_hint = Some(lang);
        self
    }

    fn diarization(mut self, d: Diarization) -> Self {
        self.diarization = d;
        self
    }

    fn word_timestamps(mut self, w: WordTimestamps) -> Self {
        self.word_timestamps = w;
        self
    }

    fn timestamps_granularity(mut self, g: TimestampsGranularity) -> Self {
        self.timestamps_granularity = g;
        self
    }

    fn punctuation(mut self, p: Punctuation) -> Self {
        self.punctuation = p;
        self
    }

    fn with_microphone(mut self, device: impl Into<String>) -> Self {
        self.source = Some(SpeechSource::Microphone {
            backend: MicBackend::Default, // TODO: use device parameter
            format: AudioFormat::Pcm16Khz,
            sample_rate: 16_000,
        });
        self
    }

    fn transcribe(mut self, path: impl Into<String>) -> Self {
        self.source = Some(SpeechSource::File {
            path: path.into(),
            format: AudioFormat::Mp3Khz44_128, // TODO: auto-detect format
        });
        self
    }

    fn with_progress<S: Into<String>>(mut self, template: S) -> Self {
        self.progress_template = Some(template.into());
        self
    }

    fn emit<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Transcript, VoiceError>) -> R + Send + 'static,
    {
        async move {
            let result = async {
                let mut transcript = Transcript::default();
                let mut stream = self.emit_stream();
                while let Some(chunk_result) = stream.next().await {
                    match chunk_result {
                        Ok(chunk) => transcript.push(chunk),
                        Err(e) => return Err(e),
                    }
                }
                Ok(transcript)
            }
            .await;
            matcher(result)
        }
    }

    fn collect(self) -> impl Future<Output = Result<Self::Transcript, VoiceError>> + Send {
        async move {
            let mut transcript = Transcript::default();
            let mut stream = self.emit_stream();
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => transcript.push(chunk),
                    Err(e) => return Err(e),
                }
            }
            Ok(transcript)
        }
    }

    fn collect_with<F, R>(self, handler: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Transcript, VoiceError>) -> R + Send + 'static,
    {
        async move {
            let result = self.collect().await;
            handler(result)
        }
    }

    fn as_text(self) -> impl Stream<Item = String> + Send {
        self.emit_stream().filter_map(|chunk_result| async move {
            match chunk_result {
                Ok(chunk) => Some(chunk.text),
                Err(_) => None,
            }
        })
    }

    fn listen<F, R>(self, matcher: F) -> impl Future<Output = R> + Send
    where
        F: FnOnce(Result<Self::Conversation, VoiceError>) -> R + Send + 'static,
    {
        async move {
            let conversation = WhisperConversation {
                stream: self.emit(),
            };
            matcher(Ok(conversation))
        }
    }
}

impl SttConversationExt for Whisper {
    fn builder() -> impl SttConversationBuilder {
        WhisperBuilder::new()
    }
}

impl SttEngine for Whisper {
    type Conv = WhisperBuilder;

    fn conversation(&self) -> Self::Conv {
        WhisperBuilder::new()
    }
}

/// Engine-specific STT conversation object for Whisper.
pub struct WhisperConversation {
    stream: WhisperStream,
}

impl SttConversation for WhisperConversation {
    type Stream = WhisperStream;

    fn into_stream(self) -> Self::Stream {
        self.stream
    }
}

/* --------------------------------------------------------------------
Internal: actual Whisper decoding implementation.
-------------------------------------------------------------------- */

mod internal {
    use super::*;
    use anyhow::Result;

    pub fn decode_to_stream(
        source: &SpeechSource,
        template: Option<&str>,
        tx: mpsc::UnboundedSender<TtsChunk>,
    ) -> Result<()> {
        match source {
            SpeechSource::File { path, .. } => decode_file_to_stream(path, template, tx),
            SpeechSource::Microphone { .. } => {
                // TODO: Implement microphone capture
                Err(anyhow::anyhow!("Microphone input not yet implemented"))
            }
        }
    }

    fn decode_file_to_stream(
        file: &str,
        template: Option<&str>,
        tx: mpsc::UnboundedSender<TtsChunk>,
    ) -> Result<()> {
        // Load PCM for actual decoding
        let (pcm, sr) = pcm_decode::pcm_decode(file)?;
        let total_secs = pcm.len() as f64 / sr as f64;

        // Run actual Whisper inference
        let segments = run_whisper_inference(&pcm, sr)?;

        for segment in segments {
            let chunk = TtsChunk::new(
                segment.start,
                segment.start + segment.duration,
                segment.dr.tokens,
                segment.dr.text,
                segment.dr.avg_logprob,
                segment.dr.no_speech_prob,
                segment.dr.temperature,
                segment.dr.compression_ratio,
            );
            tx.send(chunk)
                .map_err(|_| anyhow::anyhow!("Channel closed"))?;
        }
        drop(tx); // close channel so stream ends

        // Print progress once at 100%.
        if let Some(t) = template {
            eprintln!("{}", t.replace("{file}", file).replace("{percent}", "100"));
        }

        Ok(())
    }

    fn run_whisper_inference(
        pcm_data: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<crate::whisper::Segment>> {
        use crate::whisper::*;
        use candle::{Device, Tensor};
        use candle_core::utils::{cuda_is_available, metal_is_available};
        use candle_nn::VarBuilder;
        use candle_transformers::models::whisper::{self as m, Config};
        use hf_hub::{Repo, RepoType, api::tokio::Api};
        use tokenizers::Tokenizer;

        let device = if cuda_is_available() {
            Device::new_cuda(0)?
        } else if metal_is_available() {
            Device::new_metal(0)?
        } else {
            Device::Cpu
        };

        // Use base model by default
        let model_id = "openai/whisper-base".to_string();
        let revision = "main".to_string();

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

        // Load mel filters
        let mel_bytes = match config.num_mel_bins {
            80 => include_bytes!("melfilters.bytes").as_slice(),
            128 => include_bytes!("melfilters128.bytes").as_slice(),
            nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );

        // Resample if necessary
        if sample_rate != m::SAMPLE_RATE as u32 {
            anyhow::bail!(
                "input file must have a {} sampling rate, got {}",
                m::SAMPLE_RATE,
                sample_rate
            );
        }

        // Convert to mel spectrogram
        let mel = crate::whisper::audio::pcm_to_mel(&config, pcm_data, &mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
            &device,
        )?;

        // Load model
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };
        let model = Model::Normal(m::model::Whisper::load(&vb, config)?);

        // Run decoder
        let mut decoder = Decoder::new(
            model,
            tokenizer,
            299792458, // seed
            &device,
            None, // language_token (auto-detect)
            Task::Transcribe,
            true,  // timestamps
            false, // verbose
        )?;

        decoder.run(&mel)
    }
}
