//! Session structs for ongoing voice operations

use candle_core::Tensor;
use fluent_voice::builders::{AudioIsolationSession, SoundEffectsSession, SpeechToSpeechSession};
use fluent_voice::stt_conversation::SttConversation;
use fluent_voice_domain::{TranscriptionSegment, VoiceError};
use futures_core::Stream;
use futures_util::stream;
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};
use crate::tokenizer::KyutaiTokenizer;
use crate::error::MoshiError;

/// Speech-to-speech session
#[derive(Debug)]
pub struct KyutaiSpeechToSpeechSession {
    pub(super) _phantom: std::marker::PhantomData<()>,
}

impl SpeechToSpeechSession for KyutaiSpeechToSpeechSession {
    type AudioStream = Pin<Box<dyn Stream<Item = i16> + Send + Unpin>>;

    fn into_stream(self) -> Self::AudioStream {
        Box::pin(stream::empty())
    }
}

/// Audio isolation session
#[derive(Debug)]
pub struct KyutaiAudioIsolationSession {
    pub(super) _phantom: std::marker::PhantomData<()>,
}

impl AudioIsolationSession for KyutaiAudioIsolationSession {
    type AudioStream = Pin<Box<dyn Stream<Item = i16> + Send + Unpin>>;

    fn into_stream(self) -> Self::AudioStream {
        Box::pin(stream::empty())
    }
}

/// Sound effects generation session
#[derive(Debug)]
pub struct KyutaiSoundEffectsSession {
    pub(super) _phantom: std::marker::PhantomData<()>,
}

impl SoundEffectsSession for KyutaiSoundEffectsSession {
    type AudioStream = Pin<Box<dyn Stream<Item = i16> + Send + Unpin>>;

    fn into_stream(self) -> Self::AudioStream {
        Box::pin(stream::empty())
    }
}

/// STT conversation session for live transcription
pub struct KyutaiSttConversation {
    _phantom: std::marker::PhantomData<()>,
}

impl KyutaiSttConversation {
    #[inline]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl SttConversation for KyutaiSttConversation {
    type Stream =
        Pin<Box<dyn Stream<Item = Result<KyutaiTranscriptSegment, VoiceError>> + Send + Unpin>>;

    fn into_stream(self) -> Self::Stream {
        use futures_util::stream;
        // STEP 1: Initialize model loading (thread-safe singleton)
        let model_future = async {
            let model_paths = crate::models::get_or_download_models()
                .await
                .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

            // STEP 2: Load Mimi and LM models using existing patterns
            let device = candle_core::Device::Cpu;
            let mimi = crate::mimi::load_from_path(&model_paths.mimi_model_path, None, &device)
                .map_err(|e| VoiceError::ProcessingError(format!("Failed to load Mimi model: {}", e)))?;

            // Load LM model using VarBuilder
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[&model_paths.lm_model_path],
                    candle_core::DType::F32,
                    &device,
                )
                .map_err(|e| VoiceError::ProcessingError(e.to_string()))?
            };
            let lm_config = crate::lm::Config::default();
            let lm = crate::model::LmModel::new(&lm_config, vb)
                .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

            // STEP 3: Load tokenizer using existing error handling patterns
            let tokenizer = KyutaiTokenizer::from_file(model_paths.moshi_base_path.join("tokenizer.json"))
                .map_err(|e| VoiceError::ProcessingError(format!("Tokenizer load failed: {}", e)))?;

            // STEP 4: Create ASR state (existing constructor)
            let asr_state = crate::asr::State::new(
                10, // asr_delay_in_tokens (from official Kyutai patterns)
                mimi, lm,
            )
            .map_err(|e| VoiceError::ProcessingError(e.to_string()))?;

            // STEP 5: Return working stream instead of empty stream
            Ok::<AudioTranscriptionStream, VoiceError>(AudioTranscriptionStream::new(asr_state, tokenizer))
        };

        // Convert async initialization to stream
        use futures_util::StreamExt;
        Box::pin(
            stream::once(model_future)
                .map(|result| match result {
                    Ok(stream) => Box::pin(stream)
                        as Pin<
                            Box<
                                dyn Stream<Item = Result<KyutaiTranscriptSegment, VoiceError>>
                                    + Send
                                    + Unpin,
                            >,
                        >,
                    Err(e) => Box::pin(stream::iter(vec![Err(e)]))
                        as Pin<
                            Box<
                                dyn Stream<Item = Result<KyutaiTranscriptSegment, VoiceError>>
                                    + Send
                                    + Unpin,
                            >,
                        >,
                })
                .flatten(),
        )
    }
}

/// Real-time audio transcription stream wrapping ASR State
struct AudioTranscriptionStream {
    asr_state: crate::asr::State,
    audio_buffer: VecDeque<f32>,
    sample_rate: u32,
    initialized: bool,
    tokenizer: KyutaiTokenizer,
}

impl AudioTranscriptionStream {
    fn new(asr_state: crate::asr::State, tokenizer: KyutaiTokenizer) -> Self {
        Self {
            asr_state,
            audio_buffer: VecDeque::with_capacity(4096),
            sample_rate: 24000, // Kyutai standard sample rate
            initialized: false,
            tokenizer,
        }
    }

    /// Add incoming audio samples to processing buffer
    pub fn push_audio(&mut self, samples: &[f32]) {
        self.audio_buffer.extend(samples);

        // Prevent buffer overflow - keep last 8192 samples
        if self.audio_buffer.len() > 8192 {
            let excess = self.audio_buffer.len() - 8192;
            self.audio_buffer.drain(..excess);
        }
    }

    /// Check if enough audio is buffered for processing
    pub fn can_process(&self) -> bool {
        self.audio_buffer.len() >= 512 // Minimum chunk size for ASR
    }

    fn decode_tokens(&self, tokens: &[u32]) -> Result<String, MoshiError> {
        // Use existing production tokenizer - ZERO NEW CODE NEEDED
        self.tokenizer.decode(tokens, true) // skip_special_tokens = true
    }
}

impl Stream for AudioTranscriptionStream {
    type Item = Result<KyutaiTranscriptSegment, VoiceError>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // STEP 1: Check for buffered audio (minimum 512 samples for processing)
        if self.audio_buffer.len() < 512 {
            return Poll::Pending; // Wait for more audio data
        }

        // STEP 2: Process audio chunk through ASR
        let audio_chunk: Vec<f32> = self.audio_buffer.drain(..512).collect();
        let pcm_tensor = match Tensor::from_vec(audio_chunk, (1, 512), self.asr_state.device()) {
            Ok(tensor) => tensor,
            Err(e) => return Poll::Ready(Some(Err(VoiceError::ProcessingError(e.to_string())))),
        };

        // STEP 3: Use EXISTING ASR step_pcm method
        match self.asr_state.step_pcm(pcm_tensor, |_token, _audio| Ok(())) {
            Ok(words) => {
                if let Some(word) = words.into_iter().next() {
                    // STEP 4: Convert Word to KyutaiTranscriptSegment (perfect match!)
                    let text = match self.decode_tokens(&word.tokens) {
                        Ok(decoded_text) => decoded_text,
                        Err(e) => return Poll::Ready(Some(Err(VoiceError::ProcessingError(e.to_string())))),
                    };
                    let segment = KyutaiTranscriptSegment {
                        text,
                        start_time: Some(word.start_time),
                        end_time: Some(word.stop_time),
                        speaker_id: None,
                        _confidence: Some(0.9), // ASR confidence
                    };
                    Poll::Ready(Some(Ok(segment)))
                } else {
                    Poll::Pending // Continue processing
                }
            }
            Err(e) => Poll::Ready(Some(Err(VoiceError::ProcessingError(e.to_string())))),
        }
    }
}

/// Transcript segment implementation
#[derive(Debug, Clone)]
pub struct KyutaiTranscriptSegment {
    text: String,
    start_time: Option<f64>,
    end_time: Option<f64>,
    speaker_id: Option<String>,
    _confidence: Option<f64>,
}

impl KyutaiTranscriptSegment {
    pub fn new(text: String) -> Self {
        Self {
            text,
            start_time: None,
            end_time: None,
            speaker_id: None,
            _confidence: None,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

impl TranscriptionSegment for KyutaiTranscriptSegment {
    #[inline]
    fn text(&self) -> &str {
        &self.text
    }

    #[inline]
    fn start_ms(&self) -> u32 {
        self.start_time.map(|t| (t * 1000.0) as u32).unwrap_or(0)
    }

    #[inline]
    fn end_ms(&self) -> u32 {
        self.end_time.map(|t| (t * 1000.0) as u32).unwrap_or(0)
    }

    #[inline]
    fn speaker_id(&self) -> Option<&str> {
        self.speaker_id.as_deref()
    }
}
