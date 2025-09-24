use futures::StreamExt;
use livekit::webrtc::{audio_stream::native::NativeAudioStream, prelude::*};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::task::JoinHandle;

/// Configuration for the audio visualizer
#[derive(Clone)]
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

    #[allow(dead_code)]
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
    volume_multiplier: Arc<Mutex<f32>>,
}

impl Clone for AudioVisualizer {
    fn clone(&self) -> Self {
        Self {
            amplitude: self.amplitude.clone(),
            stats: self.stats.clone(),
            config: self.config.clone(),
            running: self.running.clone(),
            thread_handle: None, // JoinHandle cannot be cloned
            _rtc_track: self._rtc_track.clone(),
            volume_multiplier: self.volume_multiplier.clone(),
        }
    }
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
        let volume_multiplier = Arc::new(Mutex::new(1.0f32));

        let amplitude_clone = amplitude.clone();
        let stats_clone = stats.clone();
        let running_clone = running.clone();
        let volume_clone = volume_multiplier.clone();
        let mut audio_stream = NativeAudioStream::new(
            rtc_track.clone(),
            config.sample_rate as i32,
            config.num_channels as i32,
        );
        let _handle = rt_handle.clone();
        let smoothing = config.smoothing_factor;

        let thread_handle = tokio::spawn(async move {
            let mut smoothed_amplitude = 0.0f32;

            while running_clone.load(Ordering::Relaxed) {
                match audio_stream.next().await {
                    Some(frame) => {
                        if let Ok(rms) = Self::calculate_rms_from_samples(&frame.data) {
                            // Apply volume multiplier
                            let volume = volume_clone.lock().unwrap_or_else(|poisoned| {
                                tracing::warn!(
                                    "Volume mutex was poisoned in audio stream, recovering"
                                );
                                poisoned.into_inner()
                            });
                            let volume_adjusted_rms = rms * *volume;
                            drop(volume);

                            // Apply smoothing
                            smoothed_amplitude = smoothed_amplitude * smoothing
                                + volume_adjusted_rms * (1.0 - smoothing);

                            // Update ring buffer
                            let mut ring = amplitude_clone.lock().unwrap_or_else(|poisoned| {
                                tracing::warn!("Ring buffer mutex was poisoned, recovering");
                                poisoned.into_inner()
                            });
                            ring.push(smoothed_amplitude);

                            // Update statistics
                            let mut stats = stats_clone.lock().unwrap_or_else(|poisoned| {
                                tracing::warn!("Stats mutex was poisoned, recovering");
                                poisoned.into_inner()
                            });
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
            volume_multiplier,
        }
    }

    /// Calculate RMS amplitude from audio frame data (bytes)
    #[allow(dead_code)]
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

    /// Calculate RMS amplitude from i16 samples (used with NativeAudioStream)
    fn calculate_rms_from_samples(samples: &[i16]) -> Result<f32, &'static str> {
        if samples.is_empty() {
            return Ok(0.0);
        }

        let mut sum_sqr = 0.0;
        for &sample in samples {
            let val_f = sample as f32 / 32768.0;
            sum_sqr += val_f * val_f;
        }

        Ok((sum_sqr / samples.len() as f32).sqrt())
    }

    /// Get the current amplitude history
    pub fn get_amplitudes(&self) -> Vec<f32> {
        let ring = self.amplitude.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Ring buffer mutex was poisoned during get_amplitudes, recovering");
            poisoned.into_inner()
        });
        ring.as_vec()
    }

    /// Get the latest amplitude value
    pub fn get_current_amplitude(&self) -> Option<f32> {
        let ring = self.amplitude.lock().unwrap_or_else(|poisoned| {
            tracing::warn!(
                "Ring buffer mutex was poisoned during get_current_amplitude, recovering"
            );
            poisoned.into_inner()
        });
        ring.latest()
    }

    /// Get current audio statistics
    pub fn get_stats(&self) -> AudioStats {
        let stats = self.stats.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Stats mutex was poisoned during get_stats, recovering");
            poisoned.into_inner()
        });
        *stats
    }

    /// Reset peak amplitude tracking
    pub fn reset_peak(&self) {
        let mut stats = self.stats.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Stats mutex was poisoned during reset_peak, recovering");
            poisoned.into_inner()
        });
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
            handle.abort();
        }
    }

    /// Paint audio waveform visualization using egui graphics
    pub fn paint_waveform(&self, painter: &egui::Painter, rect: egui::Rect) {
        // Get current amplitude data
        let amplitudes = self.get_amplitudes();
        if amplitudes.is_empty() {
            return;
        }

        // Calculate drawing parameters
        let amplitude_count = amplitudes.len();
        let center_y = rect.center().y;
        let max_amplitude_height = rect.height() * 0.4; // Use 40% of height for amplitude range
        let step_x = rect.width() / amplitude_count.max(1) as f32;

        // Create waveform points
        let mut points = Vec::with_capacity(amplitude_count);
        for (i, &amplitude) in amplitudes.iter().enumerate() {
            let x = rect.left() + (i as f32 * step_x);
            let y = center_y - (amplitude * max_amplitude_height);
            points.push(egui::Pos2::new(x, y));
        }

        // Draw waveform as connected line segments
        let stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(0, 255, 100)); // Green waveform
        for window in points.windows(2) {
            painter.line_segment([window[0], window[1]], stroke);
        }

        // Draw center line for reference
        let center_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(100));
        painter.line_segment(
            [
                egui::Pos2::new(rect.left(), center_y),
                egui::Pos2::new(rect.right(), center_y),
            ],
            center_stroke,
        );

        // Draw current amplitude indicator (right side)
        if let Some(current_amplitude) = self.get_current_amplitude() {
            let indicator_x = rect.right() - 20.0;
            let indicator_y = center_y - (current_amplitude * max_amplitude_height);

            painter.circle_filled(
                egui::Pos2::new(indicator_x, indicator_y),
                4.0,
                egui::Color32::from_rgb(255, 255, 0), // Yellow indicator
            );
        }

        // Draw amplitude statistics text
        let stats = self.get_stats();
        let text_color = egui::Color32::WHITE;
        let font_id = egui::FontId::monospace(10.0);

        painter.text(
            egui::Pos2::new(rect.left() + 5.0, rect.top() + 5.0),
            egui::Align2::LEFT_TOP,
            format!("Cur: {:.3}", stats.current_amplitude),
            font_id.clone(),
            text_color,
        );

        painter.text(
            egui::Pos2::new(rect.left() + 5.0, rect.top() + 20.0),
            egui::Align2::LEFT_TOP,
            format!("Peak: {:.3}", stats.peak_amplitude),
            font_id.clone(),
            text_color,
        );

        painter.text(
            egui::Pos2::new(rect.left() + 5.0, rect.top() + 35.0),
            egui::Align2::LEFT_TOP,
            format!("Avg: {:.3}", stats.average_amplitude),
            font_id,
            text_color,
        );
    }

    /// Set volume multiplier for visualization amplitude (0.0 to 2.0)
    /// Based on TODO1.md requirements
    pub fn set_volume(&self, volume: f32) -> Result<(), &'static str> {
        let clamped = volume.clamp(0.0, 2.0);
        match self.volume_multiplier.lock() {
            Ok(mut vol) => {
                *vol = clamped;
                tracing::debug!("AudioVisualizer volume set to: {:.2}", clamped);
                Ok(())
            }
            Err(_) => Err("Failed to acquire volume lock"),
        }
    }

    /// Get current volume multiplier
    pub fn get_volume(&self) -> f32 {
        self.volume_multiplier
            .lock()
            .unwrap_or_else(|poisoned| {
                tracing::warn!("Volume mutex was poisoned, recovering");
                poisoned.into_inner()
            })
            .clone()
    }

    /// Mute visualization (set volume to 0)
    pub fn mute(&self) -> Result<(), &'static str> {
        self.set_volume(0.0)
    }

    /// Unmute visualization (restore to 100% volume)
    pub fn unmute(&self) -> Result<(), &'static str> {
        self.set_volume(1.0)
    }
}

impl Drop for AudioVisualizer {
    fn drop(&mut self) {
        self.stop();
    }
}
