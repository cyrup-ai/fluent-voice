//! Streaming audio output and processing components

use super::error::SpeechGenerationError;
use super::voice_params::VoiceParameters;

/// Generation chunk size for streaming
const GENERATION_CHUNK_SIZE: usize = 512;

/// Streaming audio output with zero-copy access
#[derive(Debug)]
pub struct AudioStream<'a> {
    /// Audio data reference
    data: &'a [f32],
    /// Sample rate
    sample_rate: u32,
    /// Number of channels
    channels: u8,
    /// Duration in seconds
    duration: f64,
}

impl<'a> AudioStream<'a> {
    /// Create new audio stream
    pub fn new(data: &'a [f32], sample_rate: u32, channels: u8) -> Self {
        let duration = data.len() as f64 / (sample_rate as f64 * channels as f64);
        Self {
            data,
            sample_rate,
            channels,
            duration,
        }
    }

    /// Get audio data
    #[inline]
    pub fn data(&self) -> &[f32] {
        self.data
    }

    /// Get sample rate
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get number of channels
    #[inline]
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Get duration in seconds
    #[inline]
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// Get number of samples
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if stream is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Convert to stereo (duplicate mono channel)
    pub fn to_stereo(
        &self,
        output: &mut [f32],
    ) -> std::result::Result<usize, SpeechGenerationError> {
        if self.channels == 1 {
            // Mono to stereo conversion
            let samples_to_convert = (output.len() / 2).min(self.data.len());

            for i in 0..samples_to_convert {
                output[i * 2] = self.data[i];
                output[i * 2 + 1] = self.data[i];
            }

            Ok(samples_to_convert * 2)
        } else if self.channels == 2 {
            // Already stereo, direct copy
            let samples_to_copy = output.len().min(self.data.len());
            output[..samples_to_copy].copy_from_slice(&self.data[..samples_to_copy]);
            Ok(samples_to_copy)
        } else {
            Err(SpeechGenerationError::AudioProcessing(format!(
                "Unsupported channel count: {}",
                self.channels
            )))
        }
    }

    /// Apply audio effects
    pub fn apply_effects(
        &self,
        params: &VoiceParameters,
        output: &mut [f32],
    ) -> std::result::Result<usize, SpeechGenerationError> {
        let samples_to_process = output.len().min(self.data.len());
        output[..samples_to_process].copy_from_slice(&self.data[..samples_to_process]);

        // Apply voice parameters
        let mut temp_vec = output[..samples_to_process].to_vec();
        params.apply_to_samples(&mut temp_vec);
        output[..samples_to_process].copy_from_slice(&temp_vec);

        Ok(samples_to_process)
    }
}

/// Iterator for streaming audio generation
pub struct AudioStreamIterator<'a> {
    /// Reference to the speech generator
    generator: &'a mut crate::speech_generator::SpeechGenerator,
    /// Current text position
    text_pos: usize,
    /// Text being processed
    text: String,
    /// Current generation state
    state: GenerationState,
    /// Remaining generation steps
    remaining_steps: usize,
    /// Generated audio chunks (used for buffering stream data)
    _generated_chunks: Vec<Vec<f32>>,
    /// Current chunk index (used for stream position tracking)
    _chunk_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum GenerationState {
    Initializing,
    Processing,
    #[allow(dead_code)]
    Generating,
    Finishing,
    Complete,
}

impl<'a> AudioStreamIterator<'a> {
    /// Create new streaming iterator
    pub fn new(generator: &'a mut crate::speech_generator::SpeechGenerator, text: String) -> Self {
        let remaining_steps = generator.config.max_steps;
        Self {
            generator,
            text_pos: 0,
            text,
            state: GenerationState::Initializing,
            remaining_steps,
            _generated_chunks: Vec::new(),
            _chunk_index: 0,
        }
    }
}

impl<'a> Iterator for AudioStreamIterator<'a> {
    type Item = std::result::Result<Vec<f32>, SpeechGenerationError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.state == GenerationState::Complete || self.remaining_steps == 0 {
            return None;
        }

        match self.state {
            GenerationState::Initializing => {
                // Initialize generation
                self.state = GenerationState::Processing;
                self.remaining_steps -= 1;

                // Generate initial chunk
                match self.generator.generate_chunk(&self.text[self.text_pos..]) {
                    Ok(chunk_data) => {
                        self.text_pos += GENERATION_CHUNK_SIZE.min(self.text.len() - self.text_pos);
                        if self.text_pos >= self.text.len() {
                            self.state = GenerationState::Finishing;
                        }
                        Some(Ok(chunk_data))
                    }
                    Err(e) => {
                        self.state = GenerationState::Complete;
                        Some(Err(e))
                    }
                }
            }
            GenerationState::Processing => {
                // Continue processing
                if self.text_pos < self.text.len() {
                    match self.generator.generate_chunk(&self.text[self.text_pos..]) {
                        Ok(chunk_data) => {
                            self.text_pos +=
                                GENERATION_CHUNK_SIZE.min(self.text.len() - self.text_pos);
                            self.remaining_steps -= 1;

                            if self.text_pos >= self.text.len() {
                                self.state = GenerationState::Finishing;
                            }
                            Some(Ok(chunk_data))
                        }
                        Err(e) => {
                            self.state = GenerationState::Complete;
                            Some(Err(e))
                        }
                    }
                } else {
                    self.state = GenerationState::Finishing;
                    self.next()
                }
            }
            GenerationState::Finishing => {
                // Finalize generation
                self.state = GenerationState::Complete;
                match self.generator.finalize_generation() {
                    Ok(silence_data) => Some(Ok(silence_data)),
                    Err(e) => Some(Err(e)),
                }
            }
            _ => None,
        }
    }
}
