use crate::audio_decoders::FormatInfo;
use crate::error::Error;

/// Trait for encoding PCM audio to various compressed/specialized formats
pub trait AudioFormatEncoder {
    /// Encode 16-bit PCM samples to target format
    fn encode_from_pcm(&self, pcm_data: &[i16]) -> Result<Vec<u8>, Error>;

    /// Get information about the output format
    fn get_output_format(&self) -> FormatInfo;

    /// Whether this encoder supports streaming (chunk-by-chunk processing)
    fn supports_streaming(&self) -> bool;

    /// Get optimal chunk size for streaming (if supported)
    fn get_optimal_chunk_size(&self) -> Option<usize> {
        None
    }
}

pub struct MulawEncoder {
    sample_rate: u32,
    channels: u16,
}

impl MulawEncoder {
    pub fn new() -> Self {
        Self {
            sample_rate: 8000, // Twilio standard
            channels: 1,       // Mono for telephony
        }
    }

    pub fn with_params(sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_rate,
            channels,
        }
    }

    /// Convert 16-bit linear PCM to μ-law (ITU-T G.711 standard)
    fn linear_to_mulaw(&self, linear: i16) -> u8 {
        const BIAS: i16 = 0x84;
        const CLIP: i16 = 32635;

        let mut linear = linear;

        // Extract sign bit and get absolute value
        let sign = if linear < 0 {
            linear = -linear;
            0x80
        } else {
            0x00
        };

        // Clip to maximum representable value
        if linear > CLIP {
            linear = CLIP;
        }

        // Add bias for proper quantization
        linear += BIAS;

        // Find the exponent (segment)
        let mut exponent = 7;
        for i in 0..8 {
            if linear <= (0xFF << i) {
                exponent = i;
                break;
            }
        }

        // Extract mantissa (quantization level within segment)
        let mantissa = (linear >> (exponent + 3)) & 0x0F;

        // Combine sign, exponent, and mantissa, then complement
        let mulaw = !(sign | (exponent << 4) | mantissa);

        mulaw as u8
    }
}

impl AudioFormatEncoder for MulawEncoder {
    fn encode_from_pcm(&self, pcm_data: &[i16]) -> Result<Vec<u8>, Error> {
        Ok(pcm_data
            .iter()
            .map(|&sample| self.linear_to_mulaw(sample))
            .collect())
    }

    fn get_output_format(&self) -> FormatInfo {
        FormatInfo {
            name: "μ-law".to_string(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            bit_depth: 8,
        }
    }

    fn supports_streaming(&self) -> bool {
        true // μ-law supports sample-by-sample conversion
    }

    fn get_optimal_chunk_size(&self) -> Option<usize> {
        Some(160) // 20ms at 8kHz (standard for telephony)
    }
}

impl Default for MulawEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_decoders::KnownFormatDecoder;
    use crate::shared::query_params::OutputFormat;

    #[test]
    fn test_mulaw_roundtrip_conversion() {
        let encoder = MulawEncoder::new();
        let decoder = KnownFormatDecoder::new(OutputFormat::MuLaw8000Hz);

        // Test with various PCM values
        let test_samples = vec![0, 100, -100, 1000, -1000, 32000, -32000];

        for &original in &test_samples {
            let mulaw = encoder.linear_to_mulaw(original);
            let mulaw_bytes = vec![mulaw];
            let decoded_samples = decoder.decode_to_pcm(&mulaw_bytes).unwrap();
            let decoded = decoded_samples[0];

            // μ-law has quantization error, allow small difference
            let error = (original - decoded).abs();
            assert!(
                error <= 32,
                "Roundtrip error too large: {} -> {} -> {} (error: {})",
                original,
                mulaw,
                decoded,
                error
            );
        }
    }
}
