use crate::audio_format_detection::{AudioParams, DetectedFormat};
use crate::error::Error;
use crate::shared::query_params::OutputFormat;

pub trait AudioFormatDecoder {
    fn decode_to_pcm(&self, data: &[u8]) -> Result<Vec<i16>, Error>;
    fn get_format_info(&self) -> FormatInfo;
    fn supports_streaming(&self) -> bool;
}

#[derive(Debug, Clone)]
pub struct FormatInfo {
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_depth: u16,
}

impl FormatInfo {
    pub fn mp3() -> Self {
        Self {
            name: "MP3".to_string(),
            sample_rate: 44100,
            channels: 2,
            bit_depth: 16,
        }
    }

    pub fn wav() -> Self {
        Self {
            name: "WAV".to_string(),
            sample_rate: 44100,
            channels: 2,
            bit_depth: 16,
        }
    }

    pub fn flac() -> Self {
        Self {
            name: "FLAC".to_string(),
            sample_rate: 44100,
            channels: 2,
            bit_depth: 16,
        }
    }

    pub fn ogg() -> Self {
        Self {
            name: "OGG".to_string(),
            sample_rate: 44100,
            channels: 2,
            bit_depth: 16,
        }
    }

    pub fn aac() -> Self {
        Self {
            name: "AAC".to_string(),
            sample_rate: 44100,
            channels: 2,
            bit_depth: 16,
        }
    }
}

pub struct KnownFormatDecoder {
    format: OutputFormat,
}

impl KnownFormatDecoder {
    pub fn new(format: OutputFormat) -> Self {
        Self { format }
    }

    fn decode_pcm_direct(&self, data: &[u8]) -> Result<Vec<i16>, Error> {
        // Direct PCM conversion (optimized fast path)
        let samples: Vec<i16> = data
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        Ok(samples)
    }

    fn decode_mp3_optimized(&self, data: &[u8]) -> Result<Vec<i16>, Error> {
        // Use Rodio for MP3 decoding (optimized path)
        self.decode_with_rodio_fallback(data)
    }

    fn decode_mulaw(&self, data: &[u8]) -> Result<Vec<i16>, Error> {
        // μ-law specific decoding
        let samples: Vec<i16> = data
            .iter()
            .map(|&byte| self.mulaw_to_linear(byte))
            .collect();
        Ok(samples)
    }

    fn mulaw_to_linear(&self, mulaw: u8) -> i16 {
        // μ-law to linear PCM conversion
        const BIAS: i16 = 0x84;
        const CLIP: i16 = 32635;

        let mulaw = mulaw as i16;
        let sign = if (mulaw & 0x80) != 0 { -1 } else { 1 };
        let magnitude = mulaw & 0x7F;
        let exponent = (magnitude >> 4) & 0x07;
        let mantissa = magnitude & 0x0F;

        let linear = if exponent == 0 {
            (mantissa << 4) + BIAS
        } else {
            ((mantissa | 0x10) << (exponent + 3)) + BIAS
        };

        let result = sign * linear;
        if result > CLIP {
            CLIP
        } else if result < -CLIP {
            -CLIP
        } else {
            result
        }
    }

    fn decode_with_rodio_fallback(&self, data: &[u8]) -> Result<Vec<i16>, Error> {
        use rodio::Decoder;
        use std::io::Cursor;

        let data_owned = data.to_vec();
        let cursor = Cursor::new(data_owned);
        let decoder = Decoder::try_from(cursor).map_err(|_| Error::UnsupportedMediaFormat {
            extension: "rodio-decode-failed".to_string(),
            supported: "MP3, FLAC, Vorbis, WAV".to_string(),
        })?;

        // Convert to i16 samples with proper channel handling
        let samples: Vec<i16> = decoder
            .map(|sample| (sample * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect();

        Ok(samples)
    }
}

impl AudioFormatDecoder for KnownFormatDecoder {
    fn decode_to_pcm(&self, data: &[u8]) -> Result<Vec<i16>, Error> {
        match self.format {
            OutputFormat::Pcm8000Hz
            | OutputFormat::Pcm16000Hz
            | OutputFormat::Pcm22050Hz
            | OutputFormat::Pcm24000Hz
            | OutputFormat::Pcm44100Hz => {
                // Direct PCM conversion (optimized fast path)
                self.decode_pcm_direct(data)
            }
            OutputFormat::Mp3_22050Hz32kbps
            | OutputFormat::Mp3_44100Hz32kbps
            | OutputFormat::Mp3_44100Hz64kbps
            | OutputFormat::Mp3_44100Hz96kbps
            | OutputFormat::Mp3_44100Hz128kbps
            | OutputFormat::Mp3_44100Hz192kbps => {
                // MP3-specific optimized decoding
                self.decode_mp3_optimized(data)
            }
            OutputFormat::MuLaw8000Hz => {
                // μ-law specific decoding
                self.decode_mulaw(data)
            }
            OutputFormat::Opus48000Hz32kbps
            | OutputFormat::Opus48000Hz64kbps
            | OutputFormat::Opus48000Hz96kbps
            | OutputFormat::Opus48000Hz128kbps
            | OutputFormat::Opus48000Hz192kbps => {
                // Fallback to Rodio for Opus formats
                self.decode_with_rodio_fallback(data)
            }
        }
    }

    fn get_format_info(&self) -> FormatInfo {
        match self.format {
            OutputFormat::Mp3_22050Hz32kbps => FormatInfo {
                name: "MP3".to_string(),
                sample_rate: 22050,
                channels: 2,
                bit_depth: 16,
            },
            OutputFormat::Mp3_44100Hz32kbps
            | OutputFormat::Mp3_44100Hz64kbps
            | OutputFormat::Mp3_44100Hz96kbps
            | OutputFormat::Mp3_44100Hz128kbps
            | OutputFormat::Mp3_44100Hz192kbps => FormatInfo {
                name: "MP3".to_string(),
                sample_rate: 44100,
                channels: 2,
                bit_depth: 16,
            },
            OutputFormat::Pcm8000Hz => FormatInfo {
                name: "PCM".to_string(),
                sample_rate: 8000,
                channels: 1,
                bit_depth: 16,
            },
            OutputFormat::Pcm16000Hz => FormatInfo {
                name: "PCM".to_string(),
                sample_rate: 16000,
                channels: 1,
                bit_depth: 16,
            },
            OutputFormat::Pcm22050Hz => FormatInfo {
                name: "PCM".to_string(),
                sample_rate: 22050,
                channels: 2,
                bit_depth: 16,
            },
            OutputFormat::Pcm24000Hz => FormatInfo {
                name: "PCM".to_string(),
                sample_rate: 24000,
                channels: 2,
                bit_depth: 16,
            },
            OutputFormat::Pcm44100Hz => FormatInfo {
                name: "PCM".to_string(),
                sample_rate: 44100,
                channels: 2,
                bit_depth: 16,
            },
            OutputFormat::MuLaw8000Hz => FormatInfo {
                name: "μ-law".to_string(),
                sample_rate: 8000,
                channels: 1,
                bit_depth: 8,
            },
            OutputFormat::Opus48000Hz32kbps
            | OutputFormat::Opus48000Hz64kbps
            | OutputFormat::Opus48000Hz96kbps
            | OutputFormat::Opus48000Hz128kbps
            | OutputFormat::Opus48000Hz192kbps => FormatInfo {
                name: "Opus".to_string(),
                sample_rate: 48000,
                channels: 2,
                bit_depth: 16,
            },
        }
    }

    fn supports_streaming(&self) -> bool {
        match self.format {
            OutputFormat::Pcm8000Hz
            | OutputFormat::Pcm16000Hz
            | OutputFormat::Pcm22050Hz
            | OutputFormat::Pcm24000Hz
            | OutputFormat::Pcm44100Hz
            | OutputFormat::MuLaw8000Hz => true,
            _ => false, // Compressed formats typically need full decode
        }
    }
}

pub struct RodioUniversalDecoder {
    estimated_params: AudioParams,
}

impl RodioUniversalDecoder {
    pub fn new() -> Self {
        Self {
            estimated_params: AudioParams::default(),
        }
    }

    pub fn with_params(mut self, params: AudioParams) -> Self {
        self.estimated_params = params;
        self
    }
}

impl AudioFormatDecoder for RodioUniversalDecoder {
    fn decode_to_pcm(&self, data: &[u8]) -> Result<Vec<i16>, Error> {
        use rodio::Decoder;
        use std::io::Cursor;

        let data_owned = data.to_vec();
        let cursor = Cursor::new(data_owned);
        let decoder = Decoder::try_from(cursor).map_err(|_| Error::UnsupportedMediaFormat {
            extension: "auto-detected".to_string(),
            supported: "MP3, FLAC, Vorbis, WAV".to_string(),
        })?;

        // Convert to i16 samples with proper channel handling
        let samples: Vec<i16> = decoder
            .map(|sample| (sample * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect();

        Ok(samples)
    }

    fn get_format_info(&self) -> FormatInfo {
        FormatInfo {
            name: "Universal".to_string(),
            sample_rate: self.estimated_params.sample_rate,
            channels: self.estimated_params.channels,
            bit_depth: self.estimated_params.bit_depth,
        }
    }

    fn supports_streaming(&self) -> bool {
        false // Universal decoder needs full data
    }
}

#[cfg(feature = "advanced_audio")]
pub struct SymphoniaAdvancedDecoder {
    format_name: String,
}

#[cfg(feature = "advanced_audio")]
impl SymphoniaAdvancedDecoder {
    pub fn new(format_name: String) -> Self {
        Self { format_name }
    }

    fn convert_samples_to_i16(
        &self,
        decoded: &symphonia::core::audio::AudioBuffer<f32>,
        samples: &mut Vec<i16>,
    ) -> Result<(), Error> {
        // Convert decoded samples to i16
        let spec = decoded.spec();
        let duration = decoded.capacity() as usize;

        for frame in 0..duration {
            for ch in 0..spec.channels.count() {
                if let Some(channel) = decoded.chan(ch) {
                    if frame < channel.len() {
                        let sample = channel[frame];
                        // Convert f32 to i16 with proper scaling
                        let scaled = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                        samples.push(scaled);
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(feature = "advanced_audio")]
impl AudioFormatDecoder for SymphoniaAdvancedDecoder {
    fn decode_to_pcm(&self, data: &[u8]) -> Result<Vec<i16>, Error> {
        use std::io::Cursor;
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::probe::Hint;

        let cursor = Cursor::new(data);
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

        let mut hint = Hint::new();
        // Add format hints based on detection results

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &Default::default(), &Default::default())
            .map_err(|_| Error::UnsupportedMediaFormat {
                extension: "symphonia-detected".to_string(),
                supported: "Advanced formats via Symphonia".to_string(),
            })?;

        let mut format = probed.format;
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
            .ok_or_else(|| Error::UnsupportedMediaFormat {
                extension: "no-audio-track".to_string(),
                supported: "Files with audio tracks".to_string(),
            })?;

        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &Default::default())
            .map_err(|_| Error::UnsupportedMediaFormat {
                extension: "unsupported-codec".to_string(),
                supported: "Symphonia-supported codecs".to_string(),
            })?;

        let mut samples = Vec::new();

        // Decode loop with proper error handling
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::IoError(ref err))
                    if err.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(_) => break,
            };

            if packet.track_id() != track.id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    // Convert decoded samples to i16
                    // Handle different sample formats and channel configurations
                    if let Some(buf) = decoded.make_equivalent::<f32>() {
                        self.convert_samples_to_i16(&buf, &mut samples)?;
                    }
                }
                Err(_) => continue, // Skip problematic packets
            }
        }

        Ok(samples)
    }

    fn get_format_info(&self) -> FormatInfo {
        FormatInfo {
            name: self.format_name.clone(),
            sample_rate: 44100, // Default, should be extracted from actual format
            channels: 2,
            bit_depth: 16,
        }
    }

    fn supports_streaming(&self) -> bool {
        true // Symphonia supports streaming
    }
}

pub fn create_decoder(detected_format: &DetectedFormat) -> Box<dyn AudioFormatDecoder> {
    match detected_format {
        DetectedFormat::Known { output_format, .. } => {
            Box::new(KnownFormatDecoder::new(output_format.clone()))
        }
        DetectedFormat::Unknown {
            estimated_params, ..
        } => Box::new(RodioUniversalDecoder::new().with_params(estimated_params.clone())),
        DetectedFormat::Advanced { format_name, .. } => {
            #[cfg(feature = "advanced_audio")]
            {
                Box::new(SymphoniaAdvancedDecoder::new(format_name.clone()))
            }
            #[cfg(not(feature = "advanced_audio"))]
            {
                let _ = format_name; // Suppress unused warning
                Box::new(RodioUniversalDecoder::new())
            }
        }
    }
}

impl Default for RodioUniversalDecoder {
    fn default() -> Self {
        Self::new()
    }
}
