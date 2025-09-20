//! Voice pool for hot in-memory voice management

use anyhow::Result;
use candle_core::Device;

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::codec::{VoiceCodec, VoiceData};
use super::voice_builder::VoiceBuilder;

/// A pool of loaded voices for fast runtime access
pub struct VoicePool {
    codec: VoiceCodec,
    voices: Arc<RwLock<std::collections::HashMap<std::string::String, CachedVoice>>>,
    max_voices: usize,
    max_age: Duration,
}

/// A voice entry in the pool with usage tracking
struct CachedVoice {
    data: Arc<VoiceData>,
    last_used: Instant,
    use_count: u64,
}

impl VoicePool {
    /// Create a new voice pool with default settings
    pub fn new() -> Result<Self> {
        let device = candle_core::Device::Cpu;
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("dia-voice");
        Self::new_with_config(cache_dir, device)
    }

    /// Create a new voice pool with custom cache directory and device
    pub fn new_with_config(cache_dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        Ok(Self {
            codec: VoiceCodec::new(cache_dir, device)?,
            voices: Arc::new(RwLock::new(std::collections::HashMap::new())),
            max_voices: 100,                    // Default max voices in memory
            max_age: Duration::from_secs(3600), // 1 hour default
        })
    }

    /// Set the maximum number of voices to keep in memory
    pub fn set_max_voices(&mut self, max_voices: usize) {
        self.max_voices = max_voices;
    }

    /// Set the maximum age before evicting unused voices
    pub fn set_max_age(&mut self, max_age: Duration) {
        self.max_age = max_age;
    }

    /// Load a voice, using the pool if already loaded
    pub fn load_voice(
        &self,
        voice_id: &str,
        audio_path: impl AsRef<Path>,
    ) -> Result<Arc<VoiceData>> {
        // Check if already in pool
        {
            let mut voices = self
                .voices
                .write()
                .map_err(|_| anyhow::anyhow!("Voice pool lock poisoned during read"))?;
            if let Some(cached) = voices.get_mut(voice_id) {
                cached.last_used = Instant::now();
                cached.use_count += 1;
                return Ok(cached.data.clone());
            }
        }

        // Load the voice
        let voice_data = self.codec.load_voice(audio_path)?;
        let voice_arc = Arc::new(voice_data);

        // Add to pool
        {
            let mut voices = self
                .voices
                .write()
                .map_err(|_| anyhow::anyhow!("Voice pool lock poisoned during write"))?;

            // Evict old voices if needed
            self.evict_if_needed(&mut voices);

            voices.insert(
                voice_id.to_string(),
                CachedVoice {
                    data: voice_arc.clone(),
                    last_used: Instant::now(),
                    use_count: 1,
                },
            );
        }

        Ok(voice_arc)
    }

    /// Get a voice from the pool without loading
    pub fn get_voice(&self, voice_id: &str) -> Option<Arc<VoiceData>> {
        let mut voices = self.voices.write().ok()?;
        voices.get_mut(voice_id).map(|cached| {
            cached.last_used = Instant::now();
            cached.use_count += 1;
            cached.data.clone()
        })
    }

    /// Preload multiple voices into the pool
    pub fn preload_voices(&self, voices: &[(String, PathBuf)]) -> Result<()> {
        for (voice_id, path) in voices {
            self.load_voice(voice_id, path)?;
        }
        Ok(())
    }

    /// Clear all voices from the pool
    pub fn clear(&self) {
        if let Ok(mut voices) = self.voices.write() {
            voices.clear();
        }
    }

    /// Get statistics about the pool
    pub fn stats(&self) -> PoolStats {
        let voices = match self.voices.read() {
            Ok(guard) => guard,
            Err(_) => {
                // Return empty stats if lock is poisoned
                return PoolStats {
                    voice_count: 0,
                    total_use_count: 0,
                };
            }
        };
        PoolStats {
            voice_count: voices.len(),
            total_use_count: voices.values().map(|v| v.use_count).sum(),
        }
    }

    /// Create a fluent voice builder from an audio file
    pub fn voice(self: Arc<Self>, audio_path: impl AsRef<Path>) -> VoiceBuilder {
        VoiceBuilder::new(self, audio_path)
    }

    /// Evict voices if we're over the limit or they're too old
    fn evict_if_needed(
        &self,
        voices: &mut std::collections::HashMap<std::string::String, CachedVoice>,
    ) {
        let now = Instant::now();

        // Remove voices older than max_age
        voices.retain(|_, cached| now.duration_since(cached.last_used) < self.max_age);

        // If still over limit, remove least recently used
        if voices.len() >= self.max_voices {
            // Find the LRU voice
            if let Some(lru_id) = voices
                .iter()
                .min_by_key(|(_, v)| v.last_used)
                .map(|(id, _)| id.clone())
            {
                voices.remove(&lru_id);
            }
        }
    }
}

/// Statistics about the voice pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub voice_count: usize,
    pub total_use_count: u64,
}

/// Global voice pool for the application
static GLOBAL_POOL: once_cell::sync::OnceCell<VoicePool> = once_cell::sync::OnceCell::new();

/// Initialize the global voice pool
pub fn init_global_pool(cache_dir: impl AsRef<Path>, device: Device) -> Result<()> {
    let pool = VoicePool::new_with_config(cache_dir, device)?;
    GLOBAL_POOL
        .set(pool)
        .map_err(|_| anyhow::anyhow!("Global pool already initialized"))?;
    Ok(())
}

/// Get the global voice pool
pub fn global_pool() -> Result<&'static VoicePool> {
    GLOBAL_POOL
        .get()
        .ok_or_else(|| anyhow::anyhow!("Global voice pool not initialized - call init_global_pool() first"))
}
