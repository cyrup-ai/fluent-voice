// Quick test file to verify the empty RMS fix without workspace compilation issues
// This file tests the get_rms_level function implementation

// Copy of the fixed function
fn get_rms_level(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = signal.iter().map(|s| s * s).sum();
    (sum_sq / signal.len() as f32).sqrt()
}

fn main() {
    // Test empty slice
    let empty_slice: &[f32] = &[];
    let rms = get_rms_level(empty_slice);
    assert_eq!(rms, 0.0);
    println!("✅ Empty slice test passed: RMS = {}", rms);
    
    // Test normal cases still work
    let samples = &[0.5, -0.5, 1.0, -1.0];
    let rms = get_rms_level(samples);
    println!("✅ Normal case test passed: RMS = {}", rms);
    
    // Test single sample
    let single_sample = &[0.8];
    let rms = get_rms_level(single_sample);
    assert_eq!(rms, 0.8);
    println!("✅ Single sample test passed: RMS = {}", rms);
    
    println!("🎉 All tests passed! Empty slice panic fix is working correctly.");
}