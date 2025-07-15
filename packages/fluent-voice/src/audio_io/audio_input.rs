use anyhow::Result;

#[cfg(any(feature = "encodec", feature = "mimi", feature = "snac"))]
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, formats::FormatOptions, io::MediaSourceStream,
    meta::MetadataOptions, probe::Hint,
};

/// Raw audio input consisting of PCM samples and a sample rate.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioInput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioInput {
    /// Read a wav file from disk.
    pub fn read_wav(wav_path: &str) -> Result<Self> {
        let mut reader = hound::WavReader::open(wav_path)?;
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .collect::<std::result::Result<_, _>>()?,
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
                .collect::<std::result::Result<_, _>>()?,
        };
        Ok(Self {
            samples,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        })
    }

    /// Decode audio bytes using `symphonia` (requires audio codec features).
    #[cfg(any(feature = "encodec", feature = "mimi", feature = "snac"))]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let cursor = std::io::Cursor::new(bytes.to_vec());
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
        let hint = Hint::new();
        let probed = symphonia::default::get_probe().format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )?;
        let mut format = probed.format;
        let track = format
            .default_track()
            .ok_or_else(|| anyhow::anyhow!("no supported audio tracks"))?;
        let codec_params = &track.codec_params;
        let sample_rate = codec_params
            .sample_rate
            .ok_or_else(|| anyhow::anyhow!("unknown sample rate"))?;
        #[allow(clippy::cast_possible_truncation)]
        let channels = codec_params.channels.map(|c| c.count() as u16).unwrap_or(1);
        let mut decoder =
            symphonia::default::get_codecs().make(codec_params, &DecoderOptions::default())?;
        let mut samples = Vec::new();
        loop {
            match format.next_packet() {
                Ok(packet) => {
                    let decoded = decoder.decode(&packet)?;
                    let mut buf =
                        SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                    buf.copy_interleaved_ref(decoded);
                    samples.extend_from_slice(buf.samples());
                }
                Err(symphonia::core::errors::Error::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(Self {
            samples,
            sample_rate,
            channels,
        })
    }

    /// Decode audio bytes using `symphonia` (disabled - requires audio codec features).
    #[cfg(not(any(feature = "encodec", feature = "mimi", feature = "snac")))]
    pub fn from_bytes(_bytes: &[u8]) -> Result<Self> {
        Err(anyhow::anyhow!(
            "Audio decoding requires one of: encodec, mimi, or snac features"
        ))
    }

    /// Convert multi channel audio to mono by averaging channels.
    pub fn to_mono(&self) -> Vec<f32> {
        if self.channels <= 1 {
            return self.samples.clone();
        }
        let mut mono = vec![0.0; self.samples.len() / self.channels as usize];
        for (i, sample) in self.samples.iter().enumerate() {
            mono[i / self.channels as usize] += *sample;
        }
        for s in &mut mono {
            *s /= self.channels as f32;
        }
        mono
    }
}
