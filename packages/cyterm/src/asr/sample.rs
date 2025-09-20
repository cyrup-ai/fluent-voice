/// A single sample of Linear Pulse Code Modulation (LPCM) encoded audio.
///
/// Integers are between a range of -32768 to 32768.
/// Floats are between -1.0 and 1.0.
pub trait Sample: Copy + Default + Sized {
    /// Convert the sample to a float.
    fn to_f32(self) -> f32;
}

impl Sample for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}

impl Sample for i16 {
    fn to_f32(self) -> f32 {
        f32::from(self) / 32768.0
    }
}

impl Sample for i8 {
    fn to_f32(self) -> f32 {
        f32::from(self) / 32768.0
    }
}

impl Sample for u16 {
    fn to_f32(self) -> f32 {
        f32::from(self) / 32768.0
    }
}

impl Sample for u8 {
    fn to_f32(self) -> f32 {
        f32::from(self) / 32768.0
    }
}

/// Convert a slice of samples to f32 format for audio processing
pub fn samples_to_f32<T: Sample>(samples: &[T]) -> Vec<f32> {
    samples.iter().map(|&sample| sample.to_f32()).collect()
}

/// Convert interleaved multichannel samples to mono f32 format
pub fn samples_to_mono_f32<T: Sample>(samples: &[T], num_channels: usize) -> Vec<f32> {
    if num_channels == 1 {
        return samples_to_f32(samples);
    }

    let mono_len = samples.len() / num_channels;
    let mut mono_samples = Vec::with_capacity(mono_len);

    for frame in samples.chunks_exact(num_channels) {
        // Average all channels to create mono output
        let sum: f32 = frame.iter().map(|&sample| sample.to_f32()).sum();
        mono_samples.push(sum / num_channels as f32);
    }

    mono_samples
}
