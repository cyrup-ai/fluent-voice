//! High-performance circular buffer for streaming audio

use super::error::SpeechGenerationError;

/// Audio buffer size for streaming generation (16KB = ~180ms at 44.1kHz)
const AUDIO_BUFFER_SIZE: usize = 16384;

/// High-performance circular buffer for streaming audio
#[derive(Debug)]
pub struct AudioBuffer {
    /// Pre-allocated audio data buffer
    data: Box<[f32; AUDIO_BUFFER_SIZE]>,
    /// Current write position
    write_pos: usize,
    /// Current read position
    read_pos: usize,
    /// Number of samples available for reading
    available: usize,
    /// Sample rate
    sample_rate: u32,
    /// Number of channels
    channels: u8,
}

impl AudioBuffer {
    /// Create new audio buffer with pre-allocated memory
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        Self {
            data: Box::new([0.0; AUDIO_BUFFER_SIZE]),
            write_pos: 0,
            read_pos: 0,
            available: 0,
            sample_rate,
            channels,
        }
    }

    /// Write audio samples to buffer (zero-copy when possible)
    #[inline]
    pub fn write_samples(&mut self, samples: &[f32]) -> Result<usize, SpeechGenerationError> {
        if samples.len() > self.capacity() - self.available {
            return Err(SpeechGenerationError::BufferOverflow {
                requested: samples.len(),
                available: self.capacity() - self.available,
            });
        }

        let mut written = 0;
        for &sample in samples {
            if self.available >= self.capacity() {
                break;
            }

            self.data[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % AUDIO_BUFFER_SIZE;
            self.available += 1;
            written += 1;
        }

        Ok(written)
    }

    /// Read audio samples from buffer (zero-copy when possible)
    #[inline]
    pub fn read_samples(&mut self, output: &mut [f32]) -> usize {
        let to_read = output.len().min(self.available);

        for i in 0..to_read {
            output[i] = self.data[self.read_pos];
            self.read_pos = (self.read_pos + 1) % AUDIO_BUFFER_SIZE;
        }

        self.available -= to_read;
        to_read
    }

    /// Get number of samples available for reading
    #[inline]
    pub fn available(&self) -> usize {
        self.available
    }

    /// Get buffer capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        AUDIO_BUFFER_SIZE
    }

    /// Check if buffer is full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.available == AUDIO_BUFFER_SIZE
    }

    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.available == 0
    }

    /// Clear buffer
    #[inline]
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.available = 0;
    }

    /// Get audio format information
    #[inline]
    pub fn format(&self) -> (u32, u8) {
        (self.sample_rate, self.channels)
    }
}
