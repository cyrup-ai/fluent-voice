# TASK4: Replace KoffeeEngine Stub with Real Wake Word Detection

## Issue Classification
**CRITICAL ENGINE STUB - WAKE WORD NON-FUNCTIONAL**

## Problem Description
The `KoffeeEngine` in `default_engine_coordinator.rs` lines 391-405 is stubbed and always returns `None` instead of performing real wake word detection.

## Current Stubbed Implementation
```rust
pub struct KoffeeEngine {
    #[allow(dead_code)]
    initialized: bool,
}

impl KoffeeEngine {
    pub fn new() -> Result<Self, VoiceError> {
        Ok(Self { initialized: true })
    }

    pub fn detect(&mut self, _audio_data: &[u8]) -> Result<Option<WakeWordResult>, VoiceError> {
        // Placeholder implementation - actual wake word detection would happen here
        Ok(None)  // ❌ ALWAYS RETURNS NONE
    }
}
```

## Required Implementation

### Replace with Real Koffee Integration
```rust
use koffee::{KoffeeCandle, KoffeeCandleConfig};
use koffee::wakewords::WakewordModel;

/// Real wake word engine using Koffee-Candle for detection
pub struct KoffeeEngine {
    detector: KoffeeCandle,
    models_loaded: Vec<String>,
}

impl KoffeeEngine {
    /// Create new wake word engine with default models
    pub fn new() -> Result<Self, VoiceError> {
        // Initialize Koffee detector with optimal configuration
        let config = KoffeeCandleConfig::default();
        let mut detector = KoffeeCandle::new(&config)
            .map_err(|e| VoiceError::Configuration(format!("Failed to create Koffee detector: {}", e)))?;
        
        // Load default wake word models
        let mut models_loaded = Vec::new();
        
        // Load "syrup" wake word model
        if let Ok(syrup_model) = WakewordModel::load_from_file("../koffee/syrup.rpw") {
            detector.add_wakeword_model(syrup_model)
                .map_err(|e| VoiceError::Configuration(format!("Failed to add syrup model: {}", e)))?;
            models_loaded.push("syrup".to_string());
        }
        
        Ok(Self { detector, models_loaded })
    }

    /// Detect wake words with real processing
    pub fn detect(&mut self, audio_data: &[u8]) -> Result<Option<WakeWordResult>, VoiceError> {
        // Validate audio data format (16-bit PCM expected)
        if audio_data.len() % 2 != 0 {
            return Err(VoiceError::ProcessingError(
                "Audio data length must be even for 16-bit samples".to_string()
            ));
        }
        
        // Convert bytes to f32 samples for Koffee processing
        let samples: Vec<f32> = audio_data
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / i16::MAX as f32  // Normalize to [-1.0, 1.0]
            })
            .collect();
        
        // Process audio through Koffee detection pipeline
        if let Some(detection) = self.detector.process_samples(&samples) {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            
            Ok(Some(WakeWordResult {
                word: detection.name,
                confidence: detection.score,
                timestamp,
            }))
        } else {
            Ok(None)
        }
    }
}
```

## Dependencies Integration
The koffee package is already available:
```toml
# Already in Cargo.toml:84
koffee = { path = "../koffee" }
```

## Acceptance Criteria
- [ ] **Real wake word detection** - Uses production Koffee-Candle engine with ML models
- [ ] **Proper audio conversion** - Converts audio bytes to f32 samples with validation
- [ ] **Model management** - Loads and manages wake word models (.rpw files)
- [ ] **Error handling** - Maps Koffee errors to VoiceError with context
- [ ] **No unwrap/expect** - Uses proper Result error handling throughout

## 🚀 **IMPLEMENTATION PRIORITY: HIGH**
This enables real wake word detection functionality for the voice assistant system.