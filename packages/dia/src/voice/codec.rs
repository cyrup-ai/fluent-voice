//! Voice codec module - handles audio to code conversion and caching

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

use crate::audio::SAMPLE_RATE;
use crate::codec::encode_wav;

/// Manages voice encoding and caching
pub struct VoiceCodec {
    cache_dir: PathBuf,
    device: Device,
}

impl VoiceCodec {
    /// Create a new VoiceCodec with the specified cache directory
    pub fn new(cache_dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        fs::create_dir_all(&cache_dir)?;
        Ok(Self { cache_dir, device })
    }

    /// Load and encode a voice sample, using cache if available
    pub fn load_voice(&self, audio_path: impl AsRef<Path>) -> Result<VoiceData> {
        let audio_path = audio_path.as_ref();

        // Generate cache key from file path and modification time
        let cache_key = self.generate_cache_key(audio_path)?;
        let cache_path = self.cache_dir.join(&cache_key).with_extension("voice");

        // Try to load from cache first
        if let Ok(voice_data) = self.load_from_cache(&cache_path) {
            return Ok(voice_data);
        }

        // Otherwise encode the audio
        let codes = encode_wav(
            audio_path.to_str().context("Invalid path")?,
            &self.device,
            true, // compress
        )?;

        // Create voice data
        let voice_data = VoiceData {
            codes,
            sample_rate: SAMPLE_RATE,
            source_path: audio_path.to_path_buf(),
        };

        // Save to cache
        self.save_to_cache(&voice_data, &cache_path)?;

        Ok(voice_data)
    }

    /// Generate a cache key based on file path and metadata
    fn generate_cache_key(&self, path: &Path) -> Result<String> {
        let metadata = fs::metadata(path)?;
        let modified = metadata.modified()?;

        let mut hasher = Sha256::new();
        hasher.update(path.to_string_lossy().as_bytes());
        hasher.update(format!("{modified:?}").as_bytes());

        let hash = hasher.finalize();
        Ok(hash
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>())
    }

    /// Save voice data to cache
    fn save_to_cache(&self, voice_data: &VoiceData, cache_path: &Path) -> Result<()> {
        // Save tensor dimensions and data
        let shape = voice_data.codes.dims();
        let data = voice_data.codes.to_vec2::<u32>()?;

        let cache_data = CachedVoiceData {
            shape: shape.to_vec(),
            data: data.into_iter().flatten().collect(),
            sample_rate: voice_data.sample_rate,
            source_path: voice_data.source_path.clone(),
        };

        let encoded = bincode::encode_to_vec(&cache_data, bincode::config::standard())?;
        fs::write(cache_path, encoded)?;
        Ok(())
    }

    /// Load voice data from cache
    fn load_from_cache(&self, cache_path: &Path) -> Result<VoiceData> {
        let data = fs::read(cache_path)?;
        let (cached, _): (CachedVoiceData, usize) =
            bincode::decode_from_slice(&data, bincode::config::standard())?;

        // Reconstruct tensor
        let codes = Tensor::from_vec(cached.data, cached.shape.as_slice(), &self.device)?;

        Ok(VoiceData {
            codes,
            sample_rate: cached.sample_rate,
            source_path: cached.source_path,
        })
    }
}

/// Encoded voice data ready for cloning
#[derive(Clone)]
pub struct VoiceData {
    /// EnCodec codes [T, C] where T is time steps and C is channels
    pub codes: Tensor,
    /// Sample rate (always 24kHz for Dia)
    pub sample_rate: usize,
    /// Original source path for reference
    pub source_path: PathBuf,
}

impl VoiceData {
    /// Get the duration in seconds
    pub fn duration(&self) -> Result<f32> {
        let frames = self.codes.dim(0)?;
        // EnCodec uses 75Hz frame rate for 24kHz audio
        Ok(frames as f32 / 75.0)
    }

    /// Extract a segment of codes for use as audio prompt
    /// Returns codes from start_time to end_time (in seconds)
    pub fn extract_segment(&self, start_time: f32, duration: f32) -> Result<Tensor> {
        // EnCodec frame rate is 75Hz for 24kHz
        const FRAME_RATE: f32 = 75.0;

        let start_frame = (start_time * FRAME_RATE) as usize;
        let num_frames = (duration * FRAME_RATE) as usize;
        let total_frames = self.codes.dim(0)?;

        // Clamp to valid range
        let start_frame = start_frame.min(total_frames.saturating_sub(1));
        let end_frame = (start_frame + num_frames).min(total_frames);

        // Extract segment [start:end, :]
        self.codes
            .narrow(0, start_frame, end_frame - start_frame)
            .map_err(|e| anyhow::anyhow!("Failed to extract segment: {}", e))
    }
}

/// Serializable version of voice data for caching
#[derive(bincode::Encode, bincode::Decode)]
struct CachedVoiceData {
    shape: Vec<usize>,
    data: Vec<u32>,
    sample_rate: usize,
    source_path: PathBuf,
}
