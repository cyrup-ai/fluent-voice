//! Speaker PCM processing for voice cloning and identification

use super::{
    core_generator::SpeechGenerator, error::SpeechGenerationError, voice_params::SpeakerPcmConfig,
};

impl SpeechGenerator {
    /// Process speaker PCM data for voice cloning and identification
    pub(super) fn process_speaker_pcm(
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
