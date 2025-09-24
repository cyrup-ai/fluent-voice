//! Speaker PCM processing for voice cloning and identification

use super::{
    core_generator::SpeechGenerator, error::SpeechGenerationError, voice_params::SpeakerPcmConfig,
};
use rubato::{FftFixedIn, Resampler};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

/// Performance metrics for audio processing operations
#[derive(Debug, Clone)]
pub struct AudioProcessingMetrics {
    pub decode_time_ms: f64,
    pub resample_time_ms: f64,
    pub validation_time_ms: f64,
    pub tensor_conversion_time_ms: f64,
    pub mimi_encoding_time_ms: f64,
    pub total_processing_time_ms: f64,
    pub cache_hit: bool,
    pub file_size_mb: f64,
    pub sample_count: usize,
}

/// Cached audio data with LRU tracking
#[derive(Clone)]
struct CachedAudio {
    pcm_samples: Vec<f32>,
    sample_rate: u32,
    last_accessed: std::time::Instant,
    size_mb: usize,
}

/// LRU cache for decoded audio data
pub struct AudioCache {
    cache: Arc<RwLock<HashMap<std::path::PathBuf, CachedAudio>>>,
    max_size_mb: usize,
    current_size_mb: Arc<AtomicUsize>,
}

impl AudioCache {
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size_mb,
            current_size_mb: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Get cached audio or decode using provided decoder function
    pub async fn get_or_decode<F, Fut>(
        &self,
        path: &std::path::Path,
        decoder: F,
    ) -> Result<(Vec<f32>, u32), SpeechGenerationError>
    where
        F: FnOnce(&std::path::Path) -> Fut,
        Fut: std::future::Future<Output = Result<(Vec<f32>, u32), SpeechGenerationError>>,
    {
        let path_buf = path.to_path_buf();

        // Try to get from cache first
        {
            let mut cache = self.cache.write().await;
            if let Some(cached) = cache.get_mut(&path_buf) {
                cached.last_accessed = std::time::Instant::now();
                return Ok((cached.pcm_samples.clone(), cached.sample_rate));
            }
        }

        // Not in cache, decode the audio
        let (pcm_samples, sample_rate) = decoder(path).await?;

        // Calculate size and add to cache if there's room
        let size_mb = (pcm_samples.len() * std::mem::size_of::<f32>()) / (1024 * 1024);

        if size_mb <= self.max_size_mb {
            // Evict old entries if needed
            self.evict_if_needed(size_mb).await;

            let cached_audio = CachedAudio {
                pcm_samples: pcm_samples.clone(),
                sample_rate,
                last_accessed: std::time::Instant::now(),
                size_mb,
            };

            let mut cache = self.cache.write().await;
            cache.insert(path_buf, cached_audio);
            self.current_size_mb.fetch_add(size_mb, Ordering::Relaxed);
        }

        Ok((pcm_samples, sample_rate))
    }

    /// Evict least recently used entries to make room for new entry
    async fn evict_if_needed(&self, new_size_mb: usize) {
        let current_size = self.current_size_mb.load(Ordering::Relaxed);
        if current_size + new_size_mb <= self.max_size_mb {
            return;
        }

        let mut cache = self.cache.write().await;
        let mut entries: Vec<_> = cache
            .iter()
            .map(|(k, v)| (k.clone(), v.last_accessed))
            .collect();
        entries.sort_by_key(|(_, last_accessed)| *last_accessed);

        let mut freed_size = 0;
        for (path, _) in entries {
            if current_size + new_size_mb - freed_size <= self.max_size_mb {
                break;
            }

            if let Some(removed) = cache.remove(&path) {
                freed_size += removed.size_mb;
            }
        }

        self.current_size_mb
            .fetch_sub(freed_size, Ordering::Relaxed);
    }
}

/// Pool for expensive audio processing resources
pub struct ResourcePool {
    resamplers: Arc<RwLock<HashMap<(u32, u32), Arc<Mutex<FftFixedIn<f32>>>>>>,
}

impl ResourcePool {
    pub fn new() -> Self {
        Self {
            resamplers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get or create a resampler for the given sample rates
    pub async fn get_resampler(
        &self,
        from_rate: u32,
        to_rate: u32,
    ) -> Result<Arc<Mutex<FftFixedIn<f32>>>, SpeechGenerationError> {
        let key = (from_rate, to_rate);

        // Try to get existing resampler
        {
            let resamplers = self.resamplers.read().await;
            if let Some(resampler) = resamplers.get(&key) {
                return Ok(Arc::clone(resampler));
            }
        }

        // Create new resampler if not found
        let mut resamplers = self.resamplers.write().await;

        // Double-check in case another task created it while we were waiting
        if let Some(resampler) = resamplers.get(&key) {
            return Ok(Arc::clone(resampler));
        }

        const CHUNK: usize = 1024;
        const SUB_CHUNKS: usize = 2;
        let resampler = Arc::new(Mutex::new(
            FftFixedIn::<f32>::new(from_rate as usize, to_rate as usize, CHUNK, SUB_CHUNKS, 1)
                .map_err(|e| {
                    SpeechGenerationError::AudioProcessing(format!(
                        "Resampler creation failed: {}",
                        e
                    ))
                })?,
        ));
        resamplers.insert(key, Arc::clone(&resampler));
        Ok(resampler)
    }
}

/// Audio chunk processor for streaming large files
pub struct AudioChunkProcessor {
    _path: std::path::PathBuf,
    _chunk_size: usize,
    _current_position: usize,
    _total_samples: Vec<f32>,
    _sample_rate: u32,
}

impl AudioChunkProcessor {
    pub async fn new(
        path: &std::path::Path,
        chunk_size: usize,
    ) -> Result<Self, SpeechGenerationError> {
        let path_clone = path.to_owned();

        // Load the entire file for now - in a real implementation this would use streaming I/O
        let (samples, sample_rate) =
            tokio::task::spawn_blocking(move || fluent_voice_whisper::pcm_decode(&path_clone))
                .await
                .map_err(|e| {
                    SpeechGenerationError::AudioProcessing(format!("Async decode failed: {}", e))
                })?
                .map_err(|e| {
                    SpeechGenerationError::SpeakerPcmProcessing(format!(
                        "Failed to decode audio file {:?}: {}",
                        path, e
                    ))
                })?;

        Ok(Self {
            _path: path.to_path_buf(),
            _chunk_size: chunk_size,
            _current_position: 0,
            _total_samples: samples,
            _sample_rate: sample_rate,
        })
    }

    pub async fn next_chunk(&mut self) -> Result<Option<Vec<f32>>, SpeechGenerationError> {
        if self._current_position >= self._total_samples.len() {
            return Ok(None);
        }

        let end_pos = (self._current_position + self._chunk_size).min(self._total_samples.len());
        let chunk = self._total_samples[self._current_position..end_pos].to_vec();
        self._current_position = end_pos;

        Ok(Some(chunk))
    }
}

impl SpeechGenerator {
    /// Process speaker PCM data for voice cloning and identification (async optimized)
    pub async fn process_speaker_pcm_async(
        &self,
        _speaker_id: &str,
        audio_path: Option<&std::path::Path>,
        config: &SpeakerPcmConfig,
        audio_cache: &AudioCache,
        resource_pool: &ResourcePool,
    ) -> Result<Option<candle_core::Tensor>, SpeechGenerationError> {
        let total_start = std::time::Instant::now();
        let mut metrics = AudioProcessingMetrics {
            decode_time_ms: 0.0,
            resample_time_ms: 0.0,
            validation_time_ms: 0.0,
            tensor_conversion_time_ms: 0.0,
            mimi_encoding_time_ms: 0.0,
            total_processing_time_ms: 0.0,
            cache_hit: false,
            file_size_mb: 0.0,
            sample_count: 0,
        };

        // Early return if no speaker data provided
        let audio_path = match audio_path {
            Some(path) => path,
            None => return Ok(None),
        };

        // Get file size for metrics
        if let Ok(metadata) = tokio::fs::metadata(audio_path).await {
            metrics.file_size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        }

        // Check if file is large enough for streaming
        let use_streaming = config.streaming_enabled && metrics.file_size_mb > 10.0;

        if use_streaming {
            // Use streaming processing for large files
            let result = self
                .process_speaker_pcm_streaming(_speaker_id, audio_path, config)
                .await?;

            metrics.total_processing_time_ms = total_start.elapsed().as_millis() as f64;
            self.record_processing_metrics(metrics);

            return Ok(result);
        }

        // 1. Load and decode PCM data using cache and async decoder
        let decode_start = std::time::Instant::now();
        let (pcm_samples, original_sample_rate) = if config.streaming_enabled {
            // Use the dedicated async decode method for streaming scenarios
            self.simple_wav_decode_async(audio_path).await?
        } else {
            // Use cache for non-streaming scenarios
            audio_cache
                .get_or_decode(audio_path, |path| {
                    let path_clone = path.to_owned();
                    let path_for_error = path_clone.clone();
                    async move {
                        use tokio::task;

                        let result = task::spawn_blocking(move || {
                            fluent_voice_whisper::pcm_decode(&path_clone)
                        })
                        .await
                        .map_err(|e| {
                            SpeechGenerationError::AudioProcessing(format!(
                                "Async decode failed: {}",
                                e
                            ))
                        })?
                        .map_err(|e| {
                            SpeechGenerationError::SpeakerPcmProcessing(format!(
                                "Failed to decode audio file {:?}: {}",
                                path_for_error, e
                            ))
                        })?;

                        Ok::<(Vec<f32>, u32), SpeechGenerationError>(result)
                    }
                })
                .await?
        };

        // Check if this was a cache hit by timing
        metrics.cache_hit = decode_start.elapsed().as_millis() < 50; // Less than 50ms indicates cache hit
        metrics.decode_time_ms = decode_start.elapsed().as_millis() as f64;
        metrics.sample_count = pcm_samples.len();

        // 2. Validate PCM data
        let validation_start = std::time::Instant::now();
        self.validate_pcm_data(&pcm_samples, config)?;
        metrics.validation_time_ms = validation_start.elapsed().as_millis() as f64;

        // 3. Normalize and resample audio using resource pool
        let resample_start = std::time::Instant::now();
        let normalized_samples = self
            .normalize_audio_samples_async(
                &pcm_samples,
                original_sample_rate,
                config,
                resource_pool,
            )
            .await?;
        metrics.resample_time_ms = resample_start.elapsed().as_millis() as f64;

        // 4. Convert to Candle Tensor format
        let tensor_start = std::time::Instant::now();
        let tensor = self.pcm_to_tensor(&normalized_samples, config)?;
        metrics.tensor_conversion_time_ms = tensor_start.elapsed().as_millis() as f64;

        // 5. Apply Mimi encoding if needed for speaker embedding
        let mimi_start = std::time::Instant::now();
        let processed_tensor = self.apply_mimi_encoding(&tensor)?;
        metrics.mimi_encoding_time_ms = mimi_start.elapsed().as_millis() as f64;

        metrics.total_processing_time_ms = total_start.elapsed().as_millis() as f64;
        self.record_processing_metrics(metrics);

        Ok(Some(processed_tensor))
    }

    /// Process speaker PCM data for voice cloning and identification (legacy sync version)
    pub fn process_speaker_pcm(
        &self,
        _speaker_id: &str,
        audio_path: Option<&std::path::Path>,
        config: &SpeakerPcmConfig,
    ) -> Result<Option<candle_core::Tensor>, SpeechGenerationError> {
        // Early return if no speaker data provided
        let audio_path = match audio_path {
            Some(path) => path,
            None => return Ok(None),
        };

        // 1. Load and decode PCM data using whisper package's comprehensive decoder
        let (pcm_samples, original_sample_rate) = self.simple_wav_decode(audio_path)?;

        // 2. Validate PCM data
        self.validate_pcm_data(&pcm_samples, config)?;

        // 3. Normalize and resample audio
        let normalized_samples =
            self.normalize_audio_samples(&pcm_samples, original_sample_rate, config)?;

        // 4. Convert to Candle Tensor format
        let tensor = self.pcm_to_tensor(&normalized_samples, config)?;

        // 5. Apply Mimi encoding if needed for speaker embedding
        let processed_tensor = self.apply_mimi_encoding(&tensor)?;

        Ok(Some(processed_tensor))
    }

    /// Validate PCM data meets requirements
    fn validate_pcm_data(
        &self,
        samples: &[f32],
        config: &SpeakerPcmConfig,
    ) -> Result<(), SpeechGenerationError> {
        if samples.is_empty() {
            return Err(SpeechGenerationError::InvalidVoiceParameters(
                "Empty PCM samples provided".to_string(),
            ));
        }

        if samples.len() < config.min_samples {
            return Err(SpeechGenerationError::InvalidVoiceParameters(format!(
                "Insufficient samples: {} < {} (minimum)",
                samples.len(),
                config.min_samples
            )));
        }

        if samples.len() > config.max_samples {
            return Err(SpeechGenerationError::InvalidVoiceParameters(format!(
                "Too many samples: {} > {} (maximum)",
                samples.len(),
                config.max_samples
            )));
        }

        // Validate sample range [-1.0, 1.0]
        for (i, &sample) in samples.iter().enumerate() {
            if !sample.is_finite() || sample.abs() > 1.0 {
                return Err(SpeechGenerationError::InvalidVoiceParameters(format!(
                    "Invalid sample at index {}: {} (must be finite and in [-1.0, 1.0])",
                    i, sample
                )));
            }
        }

        Ok(())
    }

    /// Normalize and resample audio to target format
    fn normalize_audio_samples(
        &self,
        samples: &[f32],
        original_sample_rate: u32,
        config: &SpeakerPcmConfig,
    ) -> Result<Vec<f32>, SpeechGenerationError> {
        let mut processed_samples = samples.to_vec();

        // 1. Resample if needed (using production FFT-based resampling)
        if original_sample_rate != config.target_sample_rate {
            processed_samples = self.resample_audio_basic(
                &processed_samples,
                original_sample_rate,
                config.target_sample_rate,
            )?;
        }

        // 2. Normalize amplitude if enabled
        if config.normalization_enabled {
            let max_amplitude = processed_samples
                .iter()
                .map(|&x| x.abs())
                .fold(0.0f32, f32::max);

            if max_amplitude > 0.0 && max_amplitude != 1.0 {
                let scale_factor = 0.95 / max_amplitude; // Leave 5% headroom
                for sample in &mut processed_samples {
                    *sample *= scale_factor;
                }
            }
        }

        // 3. Ensure target length constraints
        if processed_samples.len() > config.max_samples {
            processed_samples.truncate(config.max_samples);
        }

        Ok(processed_samples)
    }

    /// Production-grade FFT-based resampling with anti-aliasing (copied from DIA)
    fn resample_audio_basic(
        &self,
        samples: &[f32],
        from_rate: u32,
        to_rate: u32,
    ) -> Result<Vec<f32>, SpeechGenerationError> {
        if from_rate == to_rate {
            return Ok(samples.to_vec());
        }

        use rubato::{FftFixedIn, Resampler};

        // Use FftFixedIn for flexibility
        const CHUNK: usize = 1024;
        const SUB_CHUNKS: usize = 2; // Number of sub-chunks for processing
        let mut resampler =
            FftFixedIn::<f32>::new(from_rate as usize, to_rate as usize, CHUNK, SUB_CHUNKS, 1)
                .map_err(|e| {
                    SpeechGenerationError::AudioProcessing(format!(
                        "Failed to create resampler: {}",
                        e
                    ))
                })?;

        // Calculate expected output capacity
        let expected_len =
            (samples.len() as f64 * to_rate as f64 / from_rate as f64).ceil() as usize;
        let mut out = Vec::with_capacity(expected_len + CHUNK);

        // Process in chunks
        let mut pos = 0;
        while pos < samples.len() {
            let end = (pos + CHUNK).min(samples.len());
            let chunk_len = end - pos;

            // Create input buffer
            let mut input_chunk = vec![0.0; CHUNK];
            input_chunk[..chunk_len].copy_from_slice(&samples[pos..end]);

            // Process this chunk
            let block = vec![input_chunk];
            let frames = resampler.process(&block, None).map_err(|e| {
                SpeechGenerationError::AudioProcessing(format!("Resampling failed: {}", e))
            })?;
            out.extend_from_slice(&frames[0]);

            pos += chunk_len;

            // For the last partial chunk, we're done
            if chunk_len < CHUNK {
                break;
            }
        }

        Ok(out)
    }

    /// Async audio decoding with non-blocking I/O
    async fn simple_wav_decode_async(
        &self,
        path: &std::path::Path,
    ) -> Result<(Vec<f32>, u32), SpeechGenerationError> {
        use tokio::task;

        // Read file asynchronously
        let path_clone = path.to_owned();
        let result = task::spawn_blocking(move || fluent_voice_whisper::pcm_decode(&path_clone))
            .await
            .map_err(|e| {
                SpeechGenerationError::AudioProcessing(format!("Async decode failed: {}", e))
            })?
            .map_err(|e| {
                SpeechGenerationError::SpeakerPcmProcessing(format!(
                    "Failed to decode audio file {:?}: {}",
                    path, e
                ))
            })?;

        tracing::debug!(
            "Decoded {} samples at {}Hz from {:?}",
            result.0.len(),
            result.1,
            path
        );

        Ok(result)
    }

    /// Normalize and resample audio using resource pool
    async fn normalize_audio_samples_async(
        &self,
        samples: &[f32],
        original_sample_rate: u32,
        config: &SpeakerPcmConfig,
        resource_pool: &ResourcePool,
    ) -> Result<Vec<f32>, SpeechGenerationError> {
        let mut processed_samples = samples.to_vec();

        // 1. Resample if needed using pooled resampler
        if original_sample_rate != config.target_sample_rate {
            processed_samples = self
                .resample_audio_pooled(
                    &processed_samples,
                    original_sample_rate,
                    config.target_sample_rate,
                    resource_pool,
                )
                .await?;
        }

        // 2. Normalize amplitude if enabled
        if config.normalization_enabled {
            let max_amplitude = processed_samples
                .iter()
                .map(|&x| x.abs())
                .fold(0.0f32, f32::max);

            if max_amplitude > 0.0 && max_amplitude != 1.0 {
                let scale_factor = 0.95 / max_amplitude; // Leave 5% headroom
                for sample in &mut processed_samples {
                    *sample *= scale_factor;
                }
            }
        }

        // 3. Ensure target length constraints
        if processed_samples.len() > config.max_samples {
            processed_samples.truncate(config.max_samples);
        }

        Ok(processed_samples)
    }

    /// Resample audio using pooled resampler resources
    async fn resample_audio_pooled(
        &self,
        samples: &[f32],
        from_rate: u32,
        to_rate: u32,
        resource_pool: &ResourcePool,
    ) -> Result<Vec<f32>, SpeechGenerationError> {
        if from_rate == to_rate {
            return Ok(samples.to_vec());
        }

        let resampler_arc = resource_pool.get_resampler(from_rate, to_rate).await?;

        // Calculate expected output capacity
        let expected_len =
            (samples.len() as f64 * to_rate as f64 / from_rate as f64).ceil() as usize;
        let mut out = Vec::with_capacity(expected_len + 1024);

        const CHUNK: usize = 1024;

        // Process in chunks
        let mut pos = 0;
        while pos < samples.len() {
            let end = (pos + CHUNK).min(samples.len());
            let chunk_len = end - pos;

            // Create input buffer
            let mut input_chunk = vec![0.0; CHUNK];
            input_chunk[..chunk_len].copy_from_slice(&samples[pos..end]);

            // Process this chunk with pooled resampler
            let frames = {
                let mut resampler = resampler_arc.lock().map_err(|_| {
                    SpeechGenerationError::AudioProcessing("Resampler mutex poisoned".to_string())
                })?;

                let block = vec![input_chunk];
                resampler.process(&block, None).map_err(|e| {
                    SpeechGenerationError::AudioProcessing(format!("Resampling failed: {}", e))
                })?
            };

            out.extend_from_slice(&frames[0]);
            pos += chunk_len;

            // For the last partial chunk, we're done
            if chunk_len < CHUNK {
                break;
            }
        }

        Ok(out)
    }

    /// Stream-based audio processing for large files
    async fn process_speaker_pcm_streaming(
        &self,
        _speaker_id: &str,
        audio_path: &std::path::Path,
        config: &SpeakerPcmConfig,
    ) -> Result<Option<candle_core::Tensor>, SpeechGenerationError> {
        // Process in chunks to avoid loading entire file
        let mut accumulated_samples = Vec::new();
        let mut chunk_processor =
            AudioChunkProcessor::new(audio_path, config.streaming_chunk_size).await?;

        while let Some(chunk) = chunk_processor.next_chunk().await? {
            // Process each chunk individually
            let processed_chunk = self.process_audio_chunk(&chunk, config)?;
            accumulated_samples.extend(processed_chunk);

            // Optional: yield to allow other tasks to run
            tokio::task::yield_now().await;
        }

        // Continue with existing tensor conversion logic
        let tensor = self.pcm_to_tensor(&accumulated_samples, config)?;
        let processed_tensor = self.apply_mimi_encoding(&tensor)?;
        Ok(Some(processed_tensor))
    }

    /// Process individual audio chunk (used by streaming)
    fn process_audio_chunk(
        &self,
        chunk: &[f32],
        config: &SpeakerPcmConfig,
    ) -> Result<Vec<f32>, SpeechGenerationError> {
        let mut processed_chunk = chunk.to_vec();

        // Apply normalization if enabled
        if config.normalization_enabled {
            let max_amplitude = processed_chunk
                .iter()
                .map(|&x| x.abs())
                .fold(0.0f32, f32::max);

            if max_amplitude > 0.0 && max_amplitude != 1.0 {
                let scale_factor = 0.95 / max_amplitude; // Leave 5% headroom
                for sample in &mut processed_chunk {
                    *sample *= scale_factor;
                }
            }
        }

        Ok(processed_chunk)
    }

    /// Record processing metrics for performance monitoring
    fn record_processing_metrics(&self, metrics: AudioProcessingMetrics) {
        tracing::info!(
            "Audio processing completed: decode={}ms, resample={}ms, validation={}ms, tensor={}ms, mimi={}ms, total={}ms, cache_hit={}, size={}MB, samples={}",
            metrics.decode_time_ms,
            metrics.resample_time_ms,
            metrics.validation_time_ms,
            metrics.tensor_conversion_time_ms,
            metrics.mimi_encoding_time_ms,
            metrics.total_processing_time_ms,
            metrics.cache_hit,
            metrics.file_size_mb,
            metrics.sample_count
        );
    }

    /// Convert PCM samples to Candle Tensor
    fn pcm_to_tensor(
        &self,
        samples: &[f32],
        config: &SpeakerPcmConfig,
    ) -> Result<candle_core::Tensor, SpeechGenerationError> {
        use candle_core::{DType, Tensor};

        // Create tensor with shape [batch_size=1, channels, samples]
        let tensor = Tensor::from_vec(
            samples.to_vec(),
            (1, config.target_channels as usize, samples.len()),
            &self.config.device, // Use configured target device
        )
        .map_err(|e| {
            SpeechGenerationError::TensorOperation(format!(
                "Failed to create tensor on device {:?}: {}",
                self.config.device, e
            ))
        })?;

        // Convert to appropriate dtype for model
        let tensor = tensor.to_dtype(DType::F32).map_err(|e| {
            SpeechGenerationError::TensorOperation(format!("Failed to convert tensor dtype: {}", e))
        })?;

        Ok(tensor)
    }

    /// Apply Mimi encoding for speaker embedding extraction
    fn apply_mimi_encoding(
        &self,
        tensor: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, SpeechGenerationError> {
        // Use the existing Mimi encoder from the TTS model instead of creating a new one
        // This ensures we use the actual loaded model weights for proper encoding
        let encoded_tensor = self.tts_model.mimi().encode(tensor).map_err(|e| {
            SpeechGenerationError::SpeakerEmbedding(format!(
                "Failed to encode audio with Mimi: {}",
                e
            ))
        })?;

        tracing::debug!(
            "Applied Mimi encoding: input shape {:?} -> output shape {:?}",
            tensor.dims(),
            encoded_tensor.dims()
        );

        Ok(encoded_tensor)
    }

    /// Decode audio file using whisper package's comprehensive PCM decoder
    fn simple_wav_decode(
        &self,
        path: &std::path::Path,
    ) -> Result<(Vec<f32>, u32), SpeechGenerationError> {
        // Use the comprehensive PCM decoder from whisper package
        // Supports F32, U8, U16, U24, U32, S8, S16, S24, S32, F64 audio formats
        use fluent_voice_whisper::pcm_decode;

        let (pcm_samples, sample_rate) = pcm_decode(path).map_err(|e| {
            SpeechGenerationError::SpeakerPcmProcessing(format!(
                "Failed to decode audio file {:?}: {}",
                path, e
            ))
        })?;

        tracing::debug!(
            "Decoded {} samples at {}Hz from {:?}",
            pcm_samples.len(),
            sample_rate,
            path
        );

        Ok((pcm_samples, sample_rate))
    }
}
