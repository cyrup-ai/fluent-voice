//! Voice processing DSP methods for audio manipulation

use super::voice_params::VoiceParameters;

impl VoiceParameters {
    /// Apply voice parameters to audio samples
    #[inline]
    pub fn apply_to_samples(&self, samples: &mut Vec<f32>) {
        // FFT imports removed - not used in current implementation

        // Apply volume adjustment
        if self.volume != 1.0 {
            for sample in samples.iter_mut() {
                *sample *= self.volume;
            }
        }

        // Apply speed modification (time-stretching without pitch change)
        if (self.speed - 1.0).abs() > f32::EPSILON {
            let stretched = self.apply_psola_stretch(samples, self.speed);
            samples.clear();
            samples.extend_from_slice(&stretched);
        }

        // Apply pitch shifting (frequency domain)
        if self.pitch.abs() > f32::EPSILON {
            // Convert semitones to frequency ratio: 2^(semitones/12)
            let pitch_ratio = 2.0_f32.powf(self.pitch / 12.0);
            let shifted = self.apply_pitch_shift_fft(samples, pitch_ratio);
            samples.clear();
            samples.extend_from_slice(&shifted);
        }

        // Apply emotion processing (spectral filtering)
        if (self.emotion - 0.5).abs() > f32::EPSILON {
            let filtered = self.apply_emotion_spectral_filter(samples, self.emotion);
            samples.clear();
            samples.extend_from_slice(&filtered);
        }

        // Apply emphasis through dynamic range compression
        if self.emphasis != 1.0 {
            let threshold = 0.7;
            let ratio = 1.0 / self.emphasis;

            for sample in samples.iter_mut() {
                let abs_sample = sample.abs();
                if abs_sample > threshold {
                    let excess = abs_sample - threshold;
                    let compressed = threshold + excess * ratio;
                    *sample = if *sample >= 0.0 {
                        compressed
                    } else {
                        -compressed
                    };
                }
            }
        }

        // Note: pause_duration is handled at generation level, not sample level
    }

    fn apply_psola_stretch(&self, samples: &[f32], speed_factor: f32) -> Vec<f32> {
        if samples.is_empty() || (speed_factor - 1.0).abs() < f32::EPSILON {
            return samples.to_vec();
        }

        const FRAME_SIZE: usize = 1024;
        const HOP_SIZE: usize = 256;

        let input_hop = (HOP_SIZE as f32 / speed_factor) as usize;
        let output_hop = HOP_SIZE;

        // Hanning window
        let window: Vec<f32> = (0..FRAME_SIZE)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (FRAME_SIZE - 1) as f32).cos())
            })
            .collect();

        let mut output = vec![0.0f32; (samples.len() as f32 * speed_factor) as usize + FRAME_SIZE];
        let mut phase_accumulator = vec![0.0f32; FRAME_SIZE / 2 + 1];
        let mut last_phase = vec![0.0f32; FRAME_SIZE / 2 + 1];

        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let ifft = planner.plan_fft_inverse(FRAME_SIZE);

        let mut input_pos = 0;
        let mut output_pos = 0;

        while input_pos + FRAME_SIZE <= samples.len() {
            // Extract and window frame
            let mut frame: Vec<f32> = samples[input_pos..input_pos + FRAME_SIZE]
                .iter()
                .zip(&window)
                .map(|(&s, &w)| s * w)
                .collect();

            // Forward FFT
            let mut spectrum = fft.make_output_vec();
            if fft.process(&mut frame, &mut spectrum).is_err() {
                break;
            }

            // Phase vocoder processing
            for (i, bin) in spectrum.iter_mut().enumerate() {
                let magnitude = bin.norm();
                let phase = bin.arg();

                // Calculate phase difference
                let mut phase_diff = phase - last_phase[i];
                last_phase[i] = phase;

                // Unwrap phase
                while phase_diff > std::f32::consts::PI {
                    phase_diff -= 2.0 * std::f32::consts::PI;
                }
                while phase_diff < -std::f32::consts::PI {
                    phase_diff += 2.0 * std::f32::consts::PI;
                }

                // Calculate true frequency
                let bin_freq = 2.0 * std::f32::consts::PI * i as f32 / FRAME_SIZE as f32;
                let true_freq = bin_freq + phase_diff / input_hop as f32;

                // Update phase accumulator
                phase_accumulator[i] += true_freq * output_hop as f32;

                // Reconstruct bin
                *bin = rustfft::num_complex::Complex::from_polar(magnitude, phase_accumulator[i]);
            }

            // Inverse FFT
            let mut output_frame = ifft.make_input_vec();
            output_frame.copy_from_slice(&spectrum);
            let mut time_frame = vec![0.0f32; FRAME_SIZE];
            if ifft.process(&mut output_frame, &mut time_frame).is_err() {
                break;
            }

            // Overlap-add with window
            for (i, &sample) in time_frame.iter().enumerate() {
                let windowed = sample * window[i];
                if output_pos + i < output.len() {
                    output[output_pos + i] += windowed;
                }
            }

            input_pos += input_hop;
            output_pos += output_hop;
        }

        output.truncate((samples.len() as f32 * speed_factor) as usize);
        output
    }

    fn apply_pitch_shift_fft(&self, samples: &[f32], pitch_ratio: f32) -> Vec<f32> {
        if samples.is_empty() || (pitch_ratio - 1.0).abs() < f32::EPSILON {
            return samples.to_vec();
        }

        const FRAME_SIZE: usize = 2048;
        const HOP_SIZE: usize = 512;

        // Hanning window
        let window: Vec<f32> = (0..FRAME_SIZE)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (FRAME_SIZE - 1) as f32).cos())
            })
            .collect();

        let mut output = vec![0.0f32; samples.len() + FRAME_SIZE];

        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let ifft = planner.plan_fft_inverse(FRAME_SIZE);

        let mut pos = 0;

        while pos + FRAME_SIZE <= samples.len() {
            // Extract and window frame
            let mut frame: Vec<f32> = samples[pos..pos + FRAME_SIZE]
                .iter()
                .zip(&window)
                .map(|(&s, &w)| s * w)
                .collect();

            // Forward FFT
            let mut spectrum = fft.make_output_vec();
            if fft.process(&mut frame, &mut spectrum).is_err() {
                break;
            }

            // Pitch shift by frequency domain shifting
            let mut shifted_spectrum =
                vec![rustfft::num_complex::Complex::new(0.0, 0.0); spectrum.len()];

            for i in 0..spectrum.len() {
                let shifted_bin = (i as f32 * pitch_ratio) as usize;
                if shifted_bin < shifted_spectrum.len() {
                    shifted_spectrum[shifted_bin] = spectrum[i];
                }
            }

            // Inverse FFT
            let mut time_frame = vec![0.0f32; FRAME_SIZE];
            if ifft
                .process(&mut shifted_spectrum, &mut time_frame)
                .is_err()
            {
                break;
            }

            // Overlap-add with window
            for (i, &sample) in time_frame.iter().enumerate() {
                let windowed = sample * window[i];
                if pos + i < output.len() {
                    output[pos + i] += windowed;
                }
            }

            pos += HOP_SIZE;
        }

        output.truncate(samples.len());
        output
    }

    fn apply_emotion_spectral_filter(&self, samples: &[f32], emotion: f32) -> Vec<f32> {
        if samples.is_empty() || (emotion - 0.5).abs() < f32::EPSILON {
            return samples.to_vec();
        }

        const FRAME_SIZE: usize = 1024;
        const HOP_SIZE: usize = 256;

        // Hanning window
        let window: Vec<f32> = (0..FRAME_SIZE)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (FRAME_SIZE - 1) as f32).cos())
            })
            .collect();

        let mut output = vec![0.0f32; samples.len() + FRAME_SIZE];

        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let ifft = planner.plan_fft_inverse(FRAME_SIZE);

        // Emotion-based spectral filtering
        // 0.0 = sad (low-pass, reduced high frequencies)
        // 0.5 = neutral (no change)
        // 1.0 = happy (enhanced harmonics, brighter)
        let filter_curve: Vec<f32> = (0..FRAME_SIZE / 2 + 1)
            .map(|i| {
                let freq_ratio = i as f32 / (FRAME_SIZE / 2) as f32;
                if emotion < 0.5 {
                    // Sad: low-pass filter
                    let cutoff = 0.3 + 0.4 * emotion * 2.0; // 0.3 to 0.7
                    if freq_ratio < cutoff {
                        1.0
                    } else {
                        (-10.0 * (freq_ratio - cutoff)).exp()
                    }
                } else {
                    // Happy: enhance harmonics
                    let brightness = (emotion - 0.5) * 2.0; // 0.0 to 1.0
                    1.0 + brightness * (freq_ratio * 2.0).min(1.0)
                }
            })
            .collect();

        let mut pos = 0;

        while pos + FRAME_SIZE <= samples.len() {
            // Extract and window frame
            let mut frame: Vec<f32> = samples[pos..pos + FRAME_SIZE]
                .iter()
                .zip(&window)
                .map(|(&s, &w)| s * w)
                .collect();

            // Forward FFT
            let mut spectrum = fft.make_output_vec();
            if fft.process(&mut frame, &mut spectrum).is_err() {
                break;
            }

            // Apply emotional spectral filter
            for (i, bin) in spectrum.iter_mut().enumerate() {
                *bin *= filter_curve[i];
            }

            // Inverse FFT
            let mut time_frame = vec![0.0f32; FRAME_SIZE];
            if ifft.process(&mut spectrum, &mut time_frame).is_err() {
                break;
            }

            // Overlap-add with window
            for (i, &sample) in time_frame.iter().enumerate() {
                let windowed = sample * window[i];
                if pos + i < output.len() {
                    output[pos + i] += windowed;
                }
            }

            pos += HOP_SIZE;
        }

        output.truncate(samples.len());
        output
    }
}
