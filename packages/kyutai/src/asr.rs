//! Automatic Speech Recognition (ASR) module for Moshi
//!
//! Provides real-time speech recognition capabilities with word-level timing information.

use crate::model::LmModel;
// use crate::mimi::Mimi; // TODO: uncomment when Mimi is implemented
// use candle::{IndexOp, Result, Tensor}; // TODO: uncomment when Mimi is implemented
use candle_transformers::generation::LogitsProcessor;

/// Word structure containing tokens and timing information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Word {
    /// Token IDs representing the word
    pub tokens: Vec<u32>,
    /// Start time in seconds
    pub start_time: f64,
    /// End time in seconds
    pub stop_time: f64,
}

/// ASR processing state
pub struct State {
    /// Delay in tokens for ASR processing
    _asr_delay_in_tokens: usize,
    /// Current text token
    _text_token: u32,
    /// Audio tokenizer for encoding PCM data
    // _audio_tokenizer: Mimi, // TODO: uncomment when Mimi is implemented
    /// Language model for text generation
    _lm: LmModel,
    /// Processing device
    _device: candle::Device,
    /// Current step index
    _step_idx: usize,
    /// Buffer for current word tokens
    _word_tokens: Vec<u32>,
    /// Last word stop time
    _last_stop_time: f64,
    /// Logits processor for sampling
    _lp: LogitsProcessor,
}

// TODO: implement State methods when Mimi is available
/*
impl State {
    /// Create a new ASR state
    pub fn new(asr_delay_in_tokens: usize, audio_tokenizer: Mimi, lm: LmModel) -> Result<Self> {
        let text_token = lm.text_start_token();
        let device = lm.device().clone();
        let mut s = Self {
            asr_delay_in_tokens,
            lm,
            audio_tokenizer,
            device,
            text_token,
            word_tokens: Vec::with_capacity(128), // Pre-allocate for typical word length
            step_idx: 0,
            last_stop_time: 0.0,
            lp: LogitsProcessor::new(42, None, None),
        };
        s.reset()?;
        Ok(s)
    }

    /// Get the processing device
    pub fn device(&self) -> &candle::Device {
        &self.device
    }

    /// Reset the ASR state
    pub fn reset(&mut self) -> Result<()> {
        self.step_idx = 0;
        self.lm.reset_state();
        self.audio_tokenizer.reset_state();
        self.word_tokens.clear();
        let text_start_token = self.lm.text_start_token();
        let audio_pad_token = self.lm.audio_pad_token();
        let text = Tensor::from_vec(vec![text_start_token], (1, 1), &self.device)?;
        let audio_token = Tensor::from_vec(vec![audio_pad_token], (1, 1), &self.device)?;
        let mut audio_tokens = Vec::with_capacity(self.lm.in_audio_codebooks());
        for _ in 0..self.lm.in_audio_codebooks() {
            audio_tokens.push(Some(audio_token.clone()));
        }
        let (_, _) = self.lm.forward(Some(text), audio_tokens)?;
        Ok(())
    }

    /// Process PCM audio data and extract words
    pub fn step_pcm<F>(&mut self, pcm: Tensor, f: F) -> Result<Vec<Word>>
    where
        F: Fn(u32, Tensor) -> Result<()>,
    {
        let audio_tokens = self.audio_tokenizer.encode_step(&pcm.into())?;
        if let Some(audio_tokens) = audio_tokens.as_option() {
            self.step_tokens(audio_tokens, f)
        } else {
            Ok(Vec::with_capacity(0))
        }
    }

    /// Process audio tokens and extract words
    pub fn step_tokens<F>(&mut self, audio_tokens: &Tensor, f: F) -> Result<Vec<Word>>
    where
        F: Fn(u32, Tensor) -> Result<()>,
    {
        let (_one, codebooks, steps) = audio_tokens.dims3()?;
        let mut words = Vec::with_capacity(steps / 4); // Heuristic pre-allocation
        let mut audio_token_vec = Vec::with_capacity(codebooks);

        for step in 0..steps {
            let step_audio = audio_tokens.narrow(2, step, 1)?;
            f(self.text_token, step_audio)?;

            audio_token_vec.clear();
            for idx in 0..codebooks {
                let token = audio_tokens.i((0, idx, step))?.reshape((1, ()))?;
                audio_token_vec.push(Some(token));
            }

            let text = if self.step_idx >= self.asr_delay_in_tokens {
                Some(Tensor::from_vec(vec![self.text_token], (1, 1), &self.device)?)
            } else {
                None
            };

            let (text_logits, _) = self.lm.forward(text, audio_token_vec.clone())?;
            self.step_idx += 1;
            let text_logits = text_logits.i((0, 0))?;
            self.text_token = self.lp.sample(&text_logits)?;

            if self.step_idx >= self.asr_delay_in_tokens {
                if self.text_token == 0 {
                    // End of word token - flush current word
                    let mut tokens = Vec::with_capacity(self.word_tokens.len());
                    tokens.append(&mut self.word_tokens);
                    let stop_time = (self.step_idx - self.asr_delay_in_tokens) as f64 / 12.5;
                    words.push(Word {
                        tokens,
                        start_time: self.last_stop_time,
                        stop_time,
                    });
                    self.last_stop_time = stop_time;
                } else if self.text_token != 3 {
                    // Regular token (not padding) - add to current word
                    self.word_tokens.push(self.text_token);
                }
            }
        }
        Ok(words)
    }
}
*/

/// Builder for creating ASR state
#[derive(Debug)]
pub struct StateBuilder {
    asr_delay_in_tokens: usize,
}

impl StateBuilder {
    /// Create a new ASR state builder
    pub fn new() -> Self {
        Self {
            asr_delay_in_tokens: 6, // Default delay
        }
    }

    /// Set the ASR delay in tokens
    pub fn asr_delay_in_tokens(mut self, delay: usize) -> Self {
        self.asr_delay_in_tokens = delay;
        self
    }

    // TODO: uncomment when Mimi is available
    // /// Build the ASR state
    // pub fn build(self, audio_tokenizer: Mimi, lm: LmModel) -> Result<State> {
    //     State::new(self.asr_delay_in_tokens, audio_tokenizer, lm)
    // }
}

impl Default for StateBuilder {
    fn default() -> Self {
        Self::new()
    }
}
