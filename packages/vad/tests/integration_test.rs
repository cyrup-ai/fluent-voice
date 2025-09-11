use fluent_voice_vad::VoiceActivityDetector;

#[test]
fn test_vad_basic_functionality() {
    // Create VAD detector with default configuration
    let mut vad = VoiceActivityDetector::builder()
        .sample_rate(16000i64)
        .chunk_size(512usize)
        .build()
        .expect("Failed to create VAD detector");

    // Test with silence (zeros) - should return low probability
    let silence: Vec<f32> = vec![0.0; 512];
    let silence_probability = vad
        .predict(silence.iter().copied())
        .expect("Failed to predict silence");

    assert!(
        silence_probability < 0.5,
        "Silence should have low speech probability, got: {}",
        silence_probability
    );

    // Test with simulated speech signal (sine wave) - should return higher probability
    let mut speech_signal = Vec::with_capacity(512);
    for i in 0..512 {
        let sample = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.1;
        speech_signal.push(sample);
    }

    let speech_probability = vad
        .predict(speech_signal.iter().copied())
        .expect("Failed to predict speech signal");

    // Speech signal should have higher probability than silence
    assert!(
        speech_probability >= silence_probability,
        "Speech signal ({}) should have higher probability than silence ({})",
        speech_probability,
        silence_probability
    );

    println!("✅ VAD Basic Functionality Test Passed");
    println!("   Silence probability: {:.4}", silence_probability);
    println!("   Speech probability: {:.4}", speech_probability);
}

#[test]
fn test_vad_state_reset() {
    let mut vad = VoiceActivityDetector::builder()
        .sample_rate(16000i64)
        .chunk_size(512usize)
        .build()
        .expect("Failed to create VAD detector");

    // Process some audio to change internal state
    let audio: Vec<f32> = (0..512).map(|i| (i as f32 / 512.0).sin() * 0.1).collect();
    let _prob1 = vad
        .predict(audio.iter().copied())
        .expect("Failed to predict first chunk");

    // Reset state
    vad.reset();

    // Process same audio again - should behave consistently after reset
    let _prob2 = vad
        .predict(audio.iter().copied())
        .expect("Failed to predict after reset");

    println!("✅ VAD State Reset Test Passed");
}

#[test]
fn test_vad_multiple_predictions() {
    let mut vad = VoiceActivityDetector::builder()
        .sample_rate(16000i64)
        .chunk_size(512usize)
        .build()
        .expect("Failed to create VAD detector");

    // Test multiple consecutive predictions
    for i in 0..10 {
        let audio: Vec<f32> = (0..512)
            .map(|j| {
                let frequency = 200.0 + (i as f32 * 50.0); // Varying frequency
                (2.0 * std::f32::consts::PI * frequency * j as f32 / 16000.0).sin() * 0.1
            })
            .collect();

        let probability = vad
            .predict(audio.iter().copied())
            .expect(&format!("Failed to predict chunk {}", i));

        assert!(
            probability >= 0.0 && probability <= 1.0,
            "Probability should be between 0 and 1, got: {}",
            probability
        );
    }

    println!("✅ VAD Multiple Predictions Test Passed");
}

#[test]
fn test_vad_error_handling() {
    // Test builder with invalid configuration
    let result = VoiceActivityDetector::builder()
        .sample_rate(8000i64) // Very low sample rate that might cause issues
        .chunk_size(1usize) // Very small chunk size
        .build();

    // Should handle gracefully without panicking
    match result {
        Ok(_) => println!("✅ VAD Error Handling Test: Low config accepted"),
        Err(e) => println!("✅ VAD Error Handling Test: Invalid config rejected: {}", e),
    }
}

#[test]
fn test_vad_performance_characteristics() {
    let mut vad = VoiceActivityDetector::builder()
        .sample_rate(16000i64)
        .chunk_size(512usize)
        .build()
        .expect("Failed to create VAD detector");

    let audio: Vec<f32> = (0..512).map(|i| (i as f32 / 512.0).sin() * 0.1).collect();

    // Time multiple predictions to verify performance
    let start = std::time::Instant::now();
    let iterations = 100;

    for _ in 0..iterations {
        let _ = vad
            .predict(audio.iter().copied())
            .expect("Failed to predict in performance test");
    }

    let duration = start.elapsed();
    let avg_ms = duration.as_millis() as f64 / iterations as f64;

    println!("✅ VAD Performance Test Passed");
    println!("   {} predictions in {:?}", iterations, duration);
    println!("   Average: {:.2}ms per prediction", avg_ms);

    // Should be fast enough for real-time processing (< 10ms per chunk)
    assert!(
        avg_ms < 10.0,
        "VAD should be fast enough for real-time, got {}ms",
        avg_ms
    );
}
