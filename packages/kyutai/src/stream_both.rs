//! Bidirectional streaming module for full-duplex dialogue capabilities
//!
//! Enables real-time conversation with simultaneous speech recognition and synthesis.

use crate::asr::{State as AsrState, Word};
use crate::error::MoshiError;
use crate::speech_generator::{GeneratorConfig, SpeechGenerator};
use candle_core::{Device, Tensor};
use futures_core::Stream;
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;
use std::time::Instant;

/// Expected sample rate for Moshi (24kHz)
const EXPECTED_SAMPLE_RATE: u32 = 24000;
/// Minimum buffer size (10ms of audio at 24kHz)
const MIN_BUFFER_SIZE: usize = 240;
/// Maximum buffer size (1 second of audio at 24kHz)
const MAX_BUFFER_SIZE: usize = 24000;

/// Configuration for bidirectional streaming
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    /// Maximum latency in milliseconds (default: 200ms for Moshi)
    pub max_latency_ms: u32,
    /// Buffer size for audio processing
    pub audio_buffer_size: usize,
    /// Enable conversation turn detection
    pub enable_turn_detection: bool,
    /// Speech generation configuration
    pub generator_config: GeneratorConfig,
    /// Maximum number of events in buffer before dropping oldest
    pub max_event_buffer_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_latency_ms: 200,
            audio_buffer_size: 4096,
            enable_turn_detection: true,
            generator_config: GeneratorConfig::default(),
            max_event_buffer_size: 1024,
        }
    }
}

impl Config {
    /// Validate configuration parameters for production safety
    pub fn validate(&self) -> Result<(), MoshiError> {
        // Validate latency constraint is reasonable for real-time processing
        if self.max_latency_ms < 10 {
            return Err(MoshiError::InvalidInput(format!(
                "max_latency_ms too low: {}ms (minimum: 10ms for stable processing)",
                self.max_latency_ms
            )));
        }

        if self.max_latency_ms > 5000 {
            return Err(MoshiError::InvalidInput(format!(
                "max_latency_ms too high: {}ms (maximum: 5000ms for responsive interaction)",
                self.max_latency_ms
            )));
        }

        // Validate audio buffer size is compatible with audio processing requirements
        if self.audio_buffer_size < MIN_BUFFER_SIZE {
            return Err(MoshiError::InvalidInput(format!(
                "audio_buffer_size too small: {} samples (minimum: {} samples for 10ms at 24kHz)",
                self.audio_buffer_size, MIN_BUFFER_SIZE
            )));
        }

        if self.audio_buffer_size > MAX_BUFFER_SIZE * 4 {
            return Err(MoshiError::InvalidInput(format!(
                "audio_buffer_size too large: {} samples (maximum: {} samples for efficient processing)",
                self.audio_buffer_size,
                MAX_BUFFER_SIZE * 4
            )));
        }

        // Validate audio buffer size is power of 2 for optimal processing
        if !self.audio_buffer_size.is_power_of_two() {
            return Err(MoshiError::InvalidInput(format!(
                "audio_buffer_size must be power of 2: {} (use 1024, 2048, 4096, etc.)",
                self.audio_buffer_size
            )));
        }

        // Validate event buffer size prevents memory exhaustion
        if self.max_event_buffer_size < 32 {
            return Err(MoshiError::InvalidInput(format!(
                "max_event_buffer_size too small: {} (minimum: 32 for smooth streaming)",
                self.max_event_buffer_size
            )));
        }

        if self.max_event_buffer_size > 1_000_000 {
            return Err(MoshiError::InvalidInput(format!(
                "max_event_buffer_size too large: {} (maximum: 1,000,000 to prevent memory exhaustion)",
                self.max_event_buffer_size
            )));
        }

        Ok(())
    }
}

/// Events generated during bidirectional streaming
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// User speech detected with word information
    UserSpeech { words: Vec<Word>, is_final: bool },
    /// Bot response generated
    BotSpeech {
        audio_data: Vec<f32>,
        text: Option<String>,
    },
    /// Conversation turn detected
    TurnBoundary { turn: ConversationTurn },
    /// Processing error occurred
    Error { error: MoshiError },
    /// Latency metrics update
    LatencyUpdate {
        asr_latency_ms: f64,
        tts_latency_ms: f64,
        conversation_latency_ms: f64,
        timestamp: f64,
    },
}

/// Conversation turn tracking
#[derive(Debug, Clone)]
pub enum ConversationTurn {
    /// User is speaking
    UserTurn { start_time: f64 },
    /// Bot is responding
    BotTurn {
        start_time: f64,
        response_to_user: bool,
    },
    /// Silence detected
    Silence { start_time: f64, duration_ms: u32 },
}

/// Bidirectional stream for full-duplex dialogue
pub struct BidirectionalStream {
    /// ASR state for processing user audio
    asr_state: AsrState,
    /// Speech generator for bot responses
    speech_generator: SpeechGenerator,
    /// Configuration
    config: Config,
    /// Event buffer for streaming output
    event_buffer: VecDeque<StreamEvent>,
    /// Current conversation turn
    current_turn: Option<ConversationTurn>,
    /// Processing device
    device: Device,
    /// Stream active flag
    is_active: bool,
    /// Session start time for accurate timestamps
    session_start: Instant,
    /// Buffer for accumulating partial tokens for streaming transcription
    partial_tokens: Vec<u32>,
    /// ASR processing latency tracking
    asr_latency_ms: f64,
    /// TTS generation latency tracking
    tts_latency_ms: f64,
    /// End-to-end conversation latency tracking
    conversation_latency_ms: f64,
    /// Last silence detection time for timeout handling
    last_silence_time: Option<Instant>,
}

impl BidirectionalStream {
    /// Create new bidirectional stream
    pub fn new(
        asr_state: AsrState,
        speech_generator: SpeechGenerator,
        config: Config,
        device: Device,
    ) -> Result<Self, MoshiError> {
        // Validate configuration before creating stream
        config.validate()?;
        Ok(Self {
            asr_state,
            speech_generator,
            config,
            event_buffer: VecDeque::new(),
            current_turn: None,
            device,
            is_active: false,
            session_start: Instant::now(),
            partial_tokens: Vec::with_capacity(128),
            asr_latency_ms: 0.0,
            tts_latency_ms: 0.0,
            conversation_latency_ms: 0.0,
            last_silence_time: None,
        })
    }

    /// Start the bidirectional stream
    pub fn start(&mut self) -> Result<(), MoshiError> {
        self.is_active = true;
        let current_time = self.session_start.elapsed().as_secs_f64();
        self.current_turn = Some(ConversationTurn::Silence {
            start_time: current_time,
            duration_ms: 0,
        });
        Ok(())
    }

    /// Add event to buffer with size management (drop oldest if full)
    #[inline]
    fn push_event(&mut self, event: StreamEvent) {
        if self.event_buffer.len() >= self.config.max_event_buffer_size {
            // Drop oldest event to maintain buffer bounds
            self.event_buffer.pop_front();
        }
        self.event_buffer.push_back(event);
    }

    /// Validate audio format compatibility
    #[inline]
    fn validate_audio_format(&self, audio_data: &[f32]) -> Result<(), MoshiError> {
        // Validate buffer size
        let buffer_len = audio_data.len();
        if buffer_len < MIN_BUFFER_SIZE {
            return Err(MoshiError::InvalidInput(format!(
                "Audio buffer too small: {} samples (minimum: {} samples for 10ms at 24kHz)",
                buffer_len, MIN_BUFFER_SIZE
            )));
        }

        if buffer_len > MAX_BUFFER_SIZE {
            return Err(MoshiError::InvalidInput(format!(
                "Audio buffer too large: {} samples (maximum: {} samples for 1s at 24kHz)",
                buffer_len, MAX_BUFFER_SIZE
            )));
        }

        // Validate buffer size is compatible with expected sample rate
        // Buffer should contain samples for a reasonable time duration (10ms to 1000ms)
        let duration_ms = (buffer_len as f64 / EXPECTED_SAMPLE_RATE as f64) * 1000.0;
        if duration_ms < 10.0 || duration_ms > 1000.0 {
            return Err(MoshiError::InvalidInput(format!(
                "Audio buffer duration invalid: {:.1}ms (expected: 10-1000ms)",
                duration_ms
            )));
        }

        // Check for invalid audio data (NaN, infinite values)
        for (i, &sample) in audio_data.iter().enumerate() {
            if !sample.is_finite() {
                return Err(MoshiError::InvalidInput(format!(
                    "Invalid audio sample at index {}: {} (must be finite)",
                    i, sample
                )));
            }

            // Check for extreme values that could cause processing issues
            if sample.abs() > 10.0 {
                return Err(MoshiError::InvalidInput(format!(
                    "Audio sample out of range at index {}: {} (expected: -10.0 to 10.0)",
                    i, sample
                )));
            }
        }

        Ok(())
    }

    /// Stop the bidirectional stream
    pub fn stop(&mut self) -> Result<(), MoshiError> {
        self.is_active = false;
        self.current_turn = None;
        self.event_buffer.clear();
        Ok(())
    }

    /// Check for silence timeout and handle conversation turn transitions
    #[inline]
    fn check_silence_timeout(&mut self) {
        if !self.config.enable_turn_detection {
            return;
        }

        let current_time = Instant::now();
        let silence_duration_ms = (self.config.max_latency_ms as f64 * 2.0) as u64; // 2x max latency as timeout

        match &self.current_turn {
            Some(ConversationTurn::UserTurn { start_time: _ }) => {
                // Check if user turn has been silent too long
                if let Some(last_silence) = self.last_silence_time {
                    if current_time.duration_since(last_silence).as_millis() as u64
                        > silence_duration_ms
                    {
                        // UserTurn → Silence transition due to timeout
                        let session_time = self.session_start.elapsed().as_secs_f64();
                        let new_turn = ConversationTurn::Silence {
                            start_time: session_time,
                            duration_ms: silence_duration_ms as u32,
                        };
                        self.current_turn = Some(new_turn.clone());
                        self.push_event(StreamEvent::TurnBoundary { turn: new_turn });
                        self.last_silence_time = Some(current_time);
                    }
                }
            }
            Some(ConversationTurn::BotTurn { start_time, .. }) => {
                // Check if bot turn should transition to silence
                let turn_duration = self.session_start.elapsed().as_secs_f64() - start_time;
                if turn_duration * 1000.0 > silence_duration_ms as f64 {
                    // BotTurn → Silence transition due to timeout
                    let session_time = self.session_start.elapsed().as_secs_f64();
                    let new_turn = ConversationTurn::Silence {
                        start_time: session_time,
                        duration_ms: 0,
                    };
                    self.current_turn = Some(new_turn.clone());
                    self.push_event(StreamEvent::TurnBoundary { turn: new_turn });
                    self.last_silence_time = Some(current_time);
                }
            }
            Some(ConversationTurn::Silence { .. }) => {
                // Update silence tracking
                self.last_silence_time = Some(current_time);
            }
            None => {
                // Initialize silence tracking
                self.last_silence_time = Some(current_time);
            }
        }
    }

    /// Process incoming audio data
    pub fn process_audio(&mut self, audio_data: &[f32]) -> Result<(), MoshiError> {
        if !self.is_active {
            return Ok(());
        }

        // Validate audio format before processing
        self.validate_audio_format(audio_data)?;

        // Convert audio data to tensor for ASR processing
        let pcm_tensor = Tensor::from_slice(audio_data, (1, audio_data.len()), &self.device)
            .map_err(|e| {
                MoshiError::TensorCreationError(format!("Failed to create PCM tensor: {}", e))
            })?;

        // Process audio through production ASR pipeline with real-time token processing
        // Measure ASR processing latency for performance monitoring
        let asr_start = Instant::now();
        let mut streaming_tokens = Vec::new();
        let words = self
            .asr_state
            .step_pcm(pcm_tensor, |token, _tensor| {
                // Collect tokens for streaming transcription feedback
                streaming_tokens.push(token);
                Ok(())
            })
            .map_err(|e| MoshiError::Audio(format!("ASR processing failed: {}", e)))?;

        // Update ASR latency tracking and enforce constraint
        self.asr_latency_ms = asr_start.elapsed().as_secs_f64() * 1000.0;
        if self.asr_latency_ms > self.config.max_latency_ms as f64 {
            let error = MoshiError::Audio(format!(
                "ASR latency exceeded constraint: {:.1}ms > {}ms",
                self.asr_latency_ms, self.config.max_latency_ms
            ));
            self.push_event(StreamEvent::Error { error });
        }

        // Emit latency metrics update for monitoring
        self.push_event(StreamEvent::LatencyUpdate {
            asr_latency_ms: self.asr_latency_ms,
            tts_latency_ms: self.tts_latency_ms,
            conversation_latency_ms: self.conversation_latency_ms,
            timestamp: self.session_start.elapsed().as_secs_f64(),
        });

        // Process streaming tokens for immediate feedback
        if !streaming_tokens.is_empty() {
            self.partial_tokens.extend(streaming_tokens);

            // Generate intermediate transcription event for real-time feedback
            // Create a partial Word from accumulated tokens for streaming display
            if !self.partial_tokens.is_empty() {
                let current_time = self.session_start.elapsed().as_secs_f64();
                let partial_word = Word {
                    tokens: self.partial_tokens.clone(),
                    start_time: current_time,
                    stop_time: current_time,
                };

                self.push_event(StreamEvent::UserSpeech {
                    words: vec![partial_word],
                    is_final: false,
                });
            }
        }

        if !words.is_empty() {
            // Clear partial tokens when we get complete words
            self.partial_tokens.clear();

            // Update end-to-end conversation latency tracking
            let current_time = self.session_start.elapsed().as_secs_f64();
            if let Some(ConversationTurn::Silence { start_time, .. }) = &self.current_turn {
                self.conversation_latency_ms = (current_time - start_time) * 1000.0;
                if self.conversation_latency_ms > self.config.max_latency_ms as f64 {
                    let error = MoshiError::Audio(format!(
                        "Conversation latency exceeded constraint: {:.1}ms > {}ms",
                        self.conversation_latency_ms, self.config.max_latency_ms
                    ));
                    self.push_event(StreamEvent::Error { error });
                }
            }

            self.push_event(StreamEvent::UserSpeech {
                words,
                is_final: true,
            });

            // Update conversation turn with complete state machine
            if self.config.enable_turn_detection {
                let current_time = self.session_start.elapsed().as_secs_f64();
                match &self.current_turn {
                    Some(ConversationTurn::Silence { .. }) => {
                        // Silence → UserTurn transition
                        let new_turn = ConversationTurn::UserTurn {
                            start_time: current_time,
                        };
                        self.current_turn = Some(new_turn.clone());
                        self.push_event(StreamEvent::TurnBoundary { turn: new_turn });
                    }
                    Some(ConversationTurn::BotTurn { .. }) => {
                        // BotTurn → UserTurn transition (interruption)
                        let new_turn = ConversationTurn::UserTurn {
                            start_time: current_time,
                        };
                        self.current_turn = Some(new_turn.clone());
                        self.push_event(StreamEvent::TurnBoundary { turn: new_turn });
                    }
                    Some(ConversationTurn::UserTurn { .. }) => {
                        // Continue in UserTurn - no transition needed
                    }
                    None => {
                        // Initialize conversation with UserTurn
                        let new_turn = ConversationTurn::UserTurn {
                            start_time: current_time,
                        };
                        self.current_turn = Some(new_turn.clone());
                        self.push_event(StreamEvent::TurnBoundary { turn: new_turn });
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate bot response
    pub fn generate_response(&mut self, text: &str) -> Result<(), MoshiError> {
        if !self.is_active {
            return Ok(());
        }

        // Generate speech using production TTS pipeline
        // Measure TTS generation latency for performance monitoring
        let tts_start = Instant::now();
        let audio_data = self
            .speech_generator
            .generate(text)
            .map_err(|e| MoshiError::Generation(format!("TTS generation failed: {}", e)))?;

        // Update TTS latency tracking and enforce constraint
        self.tts_latency_ms = tts_start.elapsed().as_secs_f64() * 1000.0;
        if self.tts_latency_ms > self.config.max_latency_ms as f64 {
            let error = MoshiError::Generation(format!(
                "TTS latency exceeded constraint: {:.1}ms > {}ms",
                self.tts_latency_ms, self.config.max_latency_ms
            ));
            self.push_event(StreamEvent::Error { error });
        }

        // Emit latency metrics update for monitoring
        self.push_event(StreamEvent::LatencyUpdate {
            asr_latency_ms: self.asr_latency_ms,
            tts_latency_ms: self.tts_latency_ms,
            conversation_latency_ms: self.conversation_latency_ms,
            timestamp: self.session_start.elapsed().as_secs_f64(),
        });

        self.push_event(StreamEvent::BotSpeech {
            audio_data,
            text: Some(text.to_string()),
        });

        // Update conversation turn with complete state machine
        if self.config.enable_turn_detection {
            let current_time = self.session_start.elapsed().as_secs_f64();
            match &self.current_turn {
                Some(ConversationTurn::UserTurn { .. }) => {
                    // UserTurn → BotTurn transition (normal response)
                    let new_turn = ConversationTurn::BotTurn {
                        start_time: current_time,
                        response_to_user: true,
                    };
                    self.current_turn = Some(new_turn.clone());
                    self.push_event(StreamEvent::TurnBoundary { turn: new_turn });
                }
                Some(ConversationTurn::Silence { .. }) => {
                    // Silence → BotTurn transition (proactive response)
                    let new_turn = ConversationTurn::BotTurn {
                        start_time: current_time,
                        response_to_user: false,
                    };
                    self.current_turn = Some(new_turn.clone());
                    self.push_event(StreamEvent::TurnBoundary { turn: new_turn });
                }
                Some(ConversationTurn::BotTurn { .. }) => {
                    // Continue in BotTurn - no transition needed (continued response)
                }
                None => {
                    // Initialize conversation with BotTurn
                    let new_turn = ConversationTurn::BotTurn {
                        start_time: current_time,
                        response_to_user: false,
                    };
                    self.current_turn = Some(new_turn.clone());
                    self.push_event(StreamEvent::TurnBoundary { turn: new_turn });
                }
            }
        }

        Ok(())
    }
}

impl Stream for BidirectionalStream {
    type Item = Result<StreamEvent, MoshiError>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if !self.is_active {
            return Poll::Ready(None);
        }

        // Check for silence timeouts and handle conversation turn transitions
        self.check_silence_timeout();

        // Return buffered events
        if let Some(event) = self.event_buffer.pop_front() {
            return Poll::Ready(Some(Ok(event)));
        }

        // No events ready, would need to check for new audio input in real implementation
        Poll::Pending
    }
}

/// Streaming model wrapper for bidirectional processing
pub struct StreamingModel {
    /// Bidirectional stream
    stream: BidirectionalStream,
    /// Model configuration
    config: Config,
}

impl StreamingModel {
    /// Create new streaming model
    pub fn new(
        asr_state: AsrState,
        speech_generator: SpeechGenerator,
        config: Config,
        device: Device,
    ) -> Result<Self, MoshiError> {
        // Validate configuration before creating streaming model
        config.validate()?;
        let stream = BidirectionalStream::new(asr_state, speech_generator, config.clone(), device)?;
        Ok(Self { stream, config })
    }

    /// Get mutable reference to the stream
    pub fn stream_mut(&mut self) -> &mut BidirectionalStream {
        &mut self.stream
    }

    /// Get configuration
    pub fn config(&self) -> &Config {
        &self.config
    }
}
