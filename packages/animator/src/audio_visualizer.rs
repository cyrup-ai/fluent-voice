use futures::StreamExt;
use livekit::webrtc::{audio_stream::native::NativeAudioStream, prelude::*};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Configuration for the audio visualizer
pub struct AudioVisualizerConfig {
    /// Size of the amplitude history buffer
    pub buffer_size: usize,
    /// Sample rate for audio processing
    pub sample_rate: u32,
    /// Number of audio channels
    pub num_channels: u32,
    /// Smoothing factor for amplitude values (0.0 = no smoothing, 1.0 = max smoothing)
    pub smoothing_factor: f32,
}

impl Default for AudioVisualizerConfig {
    fn default() -> Self {
        Self {
            buffer_size: 100,
            sample_rate: 48000,
            num_channels: 2,
            smoothing_factor: 0.2,
        }
    }
}

/// A fixed-size ring buffer for storing amplitude values.
struct RingBuffer<T> {
    buffer: Vec<T>,
    capacity: usize,
    next_idx: usize,
    filled: bool,
}

impl<T: Default + Copy> RingBuffer<T> {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![T::default(); capacity],
            capacity,
            next_idx: 0,
            filled: false,
        }
    }

    fn push(&mut self, value: T) {
        self.buffer[self.next_idx] = value;
        self.next_idx = (self.next_idx + 1) % self.capacity;
        if self.next_idx == 0 {
            self.filled = true;
        }
    }

    fn as_vec(&self) -> Vec<T> {
        if self.filled {
            let mut out = Vec::with_capacity(self.capacity);
            out.extend_from_slice(&self.buffer[self.next_idx..]);
            out.extend_from_slice(&self.buffer[..self.next_idx]);
            out
        } else {
            self.buffer[..self.next_idx].to_vec()
        }
    }

    fn latest(&self) -> Option<T> {
        if self.filled || self.next_idx > 0 {
            let idx = if self.next_idx == 0 {
                self.capacity - 1
            } else {
                self.next_idx - 1
            };
            Some(self.buffer[idx])
        } else {
            None
        }
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        if self.filled {
            self.capacity
        } else {
            self.next_idx
        }
    }
}

/// Audio statistics tracked by the visualizer
#[derive(Debug, Clone, Copy, Default)]
pub struct AudioStats {
    /// Current RMS amplitude
    pub current_amplitude: f32,
    /// Peak amplitude since last reset
    pub peak_amplitude: f32,
    /// Average amplitude over the buffer
    pub average_amplitude: f32,
}

pub struct AudioVisualizer {
    amplitude: Arc<Mutex<RingBuffer<f32>>>,
    stats: Arc<Mutex<AudioStats>>,
    config: AudioVisualizerConfig,
    running: Arc<AtomicBool>,
    thread_handle: Option<JoinHandle<()>>,
    _rtc_track: RtcAudioTrack,
}

impl AudioVisualizer {
    pub fn new(rt_handle: &tokio::runtime::Handle, rtc_track: RtcAudioTrack) -> Self {
        Self::with_config(rt_handle, rtc_track, AudioVisualizerConfig::default())
    }

    pub fn with_config(
        rt_handle: &tokio::runtime::Handle,
        rtc_track: RtcAudioTrack,
        config: AudioVisualizerConfig,
    ) -> Self {
        let amplitude = Arc::new(Mutex::new(RingBuffer::new(config.buffer_size)));
        let stats = Arc::new(Mutex::new(AudioStats::default()));
        let running = Arc::new(AtomicBool::new(true));

        let amplitude_clone = amplitude.clone();
        let stats_clone = stats.clone();
        let running_clone = running.clone();
        let mut audio_stream =
            NativeAudioStream::new(rtc_track.clone(), config.sample_rate, config.num_channels);
        let handle = rt_handle.clone();
        let smoothing = config.smoothing_factor;

        let thread_handle = std::thread::spawn(move || {
            let mut smoothed_amplitude = 0.0f32;

            while running_clone.load(Ordering::Relaxed) {
                match handle.block_on(audio_stream.next()) {
                    Some(frame) => {
                        if let Ok(rms) = Self::calculate_rms(&frame.data) {
                            // Apply smoothing
                            smoothed_amplitude =
                                smoothed_amplitude * smoothing + rms * (1.0 - smoothing);

                            // Update ring buffer
                            let mut ring = amplitude_clone.lock();
                            ring.push(smoothed_amplitude);

                            // Update statistics
                            let mut stats = stats_clone.lock();
                            stats.current_amplitude = smoothed_amplitude;
                            stats.peak_amplitude = stats.peak_amplitude.max(smoothed_amplitude);

                            // Calculate average
                            if ring.len() > 0 {
                                let amplitudes = ring.as_vec();
                                stats.average_amplitude =
                                    amplitudes.iter().sum::<f32>() / amplitudes.len() as f32;
                            }
                        }
                    }
                    None => {
                        // Stream ended
                        break;
                    }
                }
            }
        });

        Self {
            amplitude,
            stats,
            config,
            running,
            thread_handle: Some(thread_handle),
            _rtc_track: rtc_track,
        }
    }

    /// Calculate RMS amplitude from audio frame data
    fn calculate_rms(data: &[u8]) -> Result<f32, &'static str> {
        if data.len() % 2 != 0 {
            return Err("Invalid audio data length");
        }

        let mut sum_sqr = 0.0;
        let mut count = 0;

        for sample in data.chunks_exact(2) {
            // Combine two bytes into i16 (little endian)
            let val_i16 = i16::from_le_bytes([sample[0], sample[1]]);
            let val_f = val_i16 as f32 / 32768.0;
            sum_sqr += val_f * val_f;
            count += 1;
        }

        if count > 0 {
            Ok((sum_sqr / count as f32).sqrt())
        } else {
            Ok(0.0)
        }
    }

    /// Get the current amplitude history
    pub fn get_amplitudes(&self) -> Vec<f32> {
        let ring = self.amplitude.lock();
        ring.as_vec()
    }

    /// Get the latest amplitude value
    pub fn get_current_amplitude(&self) -> Option<f32> {
        let ring = self.amplitude.lock();
        ring.latest()
    }

    /// Get current audio statistics
    pub fn get_stats(&self) -> AudioStats {
        *self.stats.lock()
    }

    /// Reset peak amplitude tracking
    pub fn reset_peak(&self) {
        let mut stats = self.stats.lock();
        stats.peak_amplitude = stats.current_amplitude;
    }

    /// Get the configuration
    pub fn config(&self) -> &AudioVisualizerConfig {
        &self.config
    }

    /// Stop the audio processing thread
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for AudioVisualizer {
    fn drop(&mut self) {
        self.stop();
    }
}
