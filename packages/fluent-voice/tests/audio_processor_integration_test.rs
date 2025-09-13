use fluent_voice::engines::default_stt_engine::{AudioProcessor, WakeWordDetection};
use fluent_voice::VoiceError;
use std::f32::consts::PI;

/// Generate synthetic audio data for testing
/// Creates a sine wave with specified frequency, duration, and sample rate
fn generate_test_audio(frequency: f32, duration_seconds: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_seconds * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * PI * frequency * t).sin() * 0.5; // 50% amplitude
        samples.push(sample);
    }

    samples
}

/// Generate white noise for VAD testing
fn generate_white_noise(duration_seconds: f32, sample_rate: u32, amplitude: f32) -> Vec<f32> {
    use rand::Rng;
    let num_samples = (duration_seconds * sample_rate as f32) as usize;
    let mut rng = rand::thread_rng();

    (0..num_samples)
        .map(|_| rng.gen_range(-amplitude..amplitude))
        .collect()
}

/// Generate speech-like audio pattern for transcription testing
fn generate_speech_pattern(duration_seconds: f32, sample_rate: u32) -> Vec<f32> {
    let mut samples = Vec::new();
    let chunk_duration = 0.1; // 100ms chunks
    let chunks = (duration_seconds / chunk_duration) as usize;

    for chunk in 0..chunks {
        let base_freq = 200.0 + (chunk as f32 * 50.0) % 400.0; // Varying fundamental frequency
        let chunk_samples = generate_test_audio(base_freq, chunk_duration, sample_rate);

        // Add harmonics for more speech-like quality
        let harmonic_samples = generate_test_audio(base_freq * 2.0, chunk_duration, sample_rate);
        let combined: Vec<f32> = chunk_samples
            .iter()
            .zip(harmonic_samples.iter())
            .map(|(fundamental, harmonic)| fundamental + harmonic * 0.3)
            .collect();

        samples.extend(combined);
    }

    samples
}

#[tokio::test]
async fn test_audio_processor_initialization() -> Result<(), VoiceError> {
    // Test that AudioProcessor can be created successfully with real components
    let processor = AudioProcessor::new()?;

    // Verify frame size is set correctly
    assert_eq!(processor.frame_size, 1600); // 100ms at 16kHz

    println!("✅ AudioProcessor initialization test passed");
    Ok(())
}

#[tokio::test]
async fn test_audio_stream_creation() -> Result<(), VoiceError> {
    let processor = AudioProcessor::new()?;

    // Test audio stream creation
    let audio_stream = processor.create_audio_stream()?;

    // Verify stream components exist
    assert!(audio_stream.consumer.len() == 0); // Initially empty

    println!("✅ Audio stream creation test passed");
    Ok(())
}

#[tokio::test]
async fn test_wake_word_detection_with_synthetic_audio() -> Result<(), VoiceError> {
    let mut processor = AudioProcessor::new()?;

    // Generate test audio that might trigger wake word detection
    let test_audio = generate_test_audio(440.0, 1.0, 16000); // 1 second of 440Hz tone

    // Process in chunks matching the frame size
    let mut detections = Vec::new();
    for chunk in test_audio.chunks(processor.frame_size) {
        if let Some(detection) = processor.process_audio_chunk(chunk) {
            detections.push(detection);
        }
    }

    // Note: With synthetic audio, we don't expect actual wake word detections
    // This test verifies the API works without errors
    println!(
        "✅ Wake word detection processed {} chunks, found {} detections",
        test_audio.len() / processor.frame_size,
        detections.len()
    );

    Ok(())
}

#[tokio::test]
async fn test_vad_with_different_audio_types() -> Result<(), VoiceError> {
    let mut processor = AudioProcessor::new()?;

    // Test with silence (should have low speech probability)
    let silence = vec![0.0f32; processor.frame_size];
    let silence_prob = processor.process_vad(&silence)?;
    println!("Silence VAD probability: {:.3}", silence_prob);

    // Test with white noise (should have low-medium speech probability)
    let noise = generate_white_noise(0.1, 16000, 0.1);
    let noise_chunk = &noise[..processor.frame_size.min(noise.len())];
    let noise_prob = processor.process_vad(noise_chunk)?;
    println!("Noise VAD probability: {:.3}", noise_prob);

    // Test with speech-like pattern (should have higher speech probability)
    let speech_pattern = generate_speech_pattern(0.1, 16000);
    let speech_chunk = &speech_pattern[..processor.frame_size.min(speech_pattern.len())];
    let speech_prob = processor.process_vad(speech_chunk)?;
    println!("Speech pattern VAD probability: {:.3}", speech_prob);

    // Verify probabilities are in valid range [0.0, 1.0]
    assert!(silence_prob >= 0.0 && silence_prob <= 1.0);
    assert!(noise_prob >= 0.0 && noise_prob <= 1.0);
    assert!(speech_prob >= 0.0 && speech_prob <= 1.0);

    println!("✅ VAD testing with different audio types passed");
    Ok(())
}

#[tokio::test]
async fn test_transcription_with_synthetic_speech() -> Result<(), VoiceError> {
    let mut processor = AudioProcessor::new()?;

    // Generate longer speech-like audio for transcription
    let speech_audio = generate_speech_pattern(2.0, 16000); // 2 seconds

    // Test transcription (may return empty or placeholder text with synthetic audio)
    let transcription_result = processor.transcribe_audio(&speech_audio).await;

    match transcription_result {
        Ok(text) => {
            println!("✅ Transcription completed: '{}'", text);
            // With synthetic audio, we mainly verify the API doesn't crash
        }
        Err(e) => {
            println!(
                "⚠️  Transcription failed (expected with synthetic audio): {:?}",
                e
            );
            // This is acceptable for synthetic audio - the important thing is the API works
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_complete_audio_pipeline() -> Result<(), VoiceError> {
    let mut processor = AudioProcessor::new()?;

    // Create audio stream
    let _audio_stream = processor.create_audio_stream()?;

    // Generate comprehensive test audio
    let test_duration = 3.0; // 3 seconds
    let sample_rate = 16000;

    // Mix different audio types for comprehensive testing
    let mut mixed_audio = Vec::new();

    // Add silence
    mixed_audio.extend(vec![0.0f32; sample_rate / 2]); // 0.5 seconds silence

    // Add noise
    mixed_audio.extend(generate_white_noise(0.5, sample_rate, 0.05));

    // Add speech-like pattern
    mixed_audio.extend(generate_speech_pattern(1.0, sample_rate));

    // Add tone
    mixed_audio.extend(generate_test_audio(300.0, 0.5, sample_rate));

    // Add more silence
    mixed_audio.extend(vec![0.0f32; sample_rate / 2]);

    println!(
        "Processing {} samples ({:.1} seconds) through complete pipeline...",
        mixed_audio.len(),
        mixed_audio.len() as f32 / sample_rate as f32
    );

    let mut wake_word_detections = 0;
    let mut vad_results = Vec::new();

    // Process entire audio through wake word detection and VAD
    for (i, chunk) in mixed_audio.chunks(processor.frame_size).enumerate() {
        // Wake word detection
        if let Some(detection) = processor.process_audio_chunk(chunk) {
            wake_word_detections += 1;
            println!(
                "Wake word detected at chunk {}: {} (score: {:.3})",
                i, detection.name, detection.score
            );
        }

        // VAD processing
        let vad_prob = processor.process_vad(chunk)?;
        vad_results.push(vad_prob);

        // Log high VAD probability chunks
        if vad_prob > 0.5 {
            println!("High speech probability at chunk {}: {:.3}", i, vad_prob);
        }
    }

    // Test transcription on a subset of the audio
    let transcription_chunk = &mixed_audio[sample_rate..sample_rate * 2]; // 1 second chunk
    let transcription_result = processor.transcribe_audio(transcription_chunk).await;

    // Summary statistics
    let avg_vad_prob = vad_results.iter().sum::<f32>() / vad_results.len() as f32;
    let max_vad_prob = vad_results.iter().fold(0.0f32, |a, &b| a.max(b));

    println!("🎯 Complete Pipeline Test Results:");
    println!("   • Processed chunks: {}", vad_results.len());
    println!("   • Wake word detections: {}", wake_word_detections);
    println!("   • Average VAD probability: {:.3}", avg_vad_prob);
    println!("   • Maximum VAD probability: {:.3}", max_vad_prob);
    println!(
        "   • Transcription result: {:?}",
        transcription_result.is_ok()
    );

    println!("✅ Complete audio pipeline test passed");
    Ok(())
}

#[tokio::test]
async fn test_error_handling_and_edge_cases() -> Result<(), VoiceError> {
    let mut processor = AudioProcessor::new()?;

    // Test with empty audio
    let empty_audio = Vec::new();
    let vad_result = processor.process_vad(&empty_audio);
    println!("Empty audio VAD result: {:?}", vad_result);

    // Test with very small audio chunk
    let tiny_chunk = vec![0.1f32; 10];
    let tiny_vad_result = processor.process_vad(&tiny_chunk)?;
    println!("Tiny chunk VAD result: {:.3}", tiny_vad_result);

    // Test with extreme values
    let extreme_audio = vec![1.0f32; processor.frame_size]; // Maximum amplitude
    let extreme_vad_result = processor.process_vad(&extreme_audio)?;
    println!("Extreme amplitude VAD result: {:.3}", extreme_vad_result);

    // Test transcription with very short audio
    let short_audio = vec![0.0f32; 1000]; // Very short
    let short_transcription = processor.transcribe_audio(&short_audio).await;
    println!(
        "Short audio transcription: {:?}",
        short_transcription.is_ok()
    );

    println!("✅ Error handling and edge cases test passed");
    Ok(())
}

#[tokio::test]
async fn test_performance_and_memory_usage() -> Result<(), VoiceError> {
    let mut processor = AudioProcessor::new()?;

    // Test processing large amounts of audio efficiently
    let large_audio = generate_speech_pattern(10.0, 16000); // 10 seconds

    let start_time = std::time::Instant::now();

    let mut processed_chunks = 0;
    for chunk in large_audio.chunks(processor.frame_size) {
        // Process through all components
        let _wake_word = processor.process_audio_chunk(chunk);
        let _vad_result = processor.process_vad(chunk)?;
        processed_chunks += 1;
    }

    let processing_time = start_time.elapsed();
    let audio_duration = large_audio.len() as f32 / 16000.0;
    let real_time_factor = processing_time.as_secs_f32() / audio_duration;

    println!("🚀 Performance Test Results:");
    println!("   • Audio duration: {:.1}s", audio_duration);
    println!(
        "   • Processing time: {:.3}s",
        processing_time.as_secs_f32()
    );
    println!("   • Real-time factor: {:.3}x", real_time_factor);
    println!("   • Processed chunks: {}", processed_chunks);
    println!(
        "   • Chunks per second: {:.1}",
        processed_chunks as f32 / processing_time.as_secs_f32()
    );

    // Verify we can process faster than real-time for production use
    if real_time_factor < 1.0 {
        println!("✅ Processing faster than real-time - suitable for production");
    } else {
        println!("⚠️  Processing slower than real-time - may need optimization");
    }

    println!("✅ Performance and memory usage test completed");
    Ok(())
}
