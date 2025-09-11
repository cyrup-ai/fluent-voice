# Cyrup Wake Word System Guide

This guide explains how to use the "cyrup" wake word and "cyrup stop" unwake command system in Koffee.

## Quick Start (Using Existing Model)

The repository already includes a trained model for "syrup" (pronounced "see-rup") that can be used for "cyrup":

```bash
# Run the basic wake word example
cargo run --example cyrup_wake

# Run the advanced example with wake/unwake support
cargo run --example test_cyrup_models
```

## Architecture

The system uses two separate models:
1. **Wake Model** (`syrup.rpw`) - Detects "cyrup" to wake the system
2. **Stop Model** (`cyrup_stop.rpw`) - Detects "cyrup stop" to sleep (optional)

## Training Custom Models

### 1. Record Training Samples

First, record samples for your wake phrase:

```bash
# Record "cyrup" samples (if you want your own pronunciation)
cargo run --example record_training_samples -- --phrase "cyrup" --output-dir cyrup_training -n 15

# Record "cyrup stop" samples  
cargo run --example record_training_samples -- --phrase "cyrup stop" --output-dir cyrup_stop_training -n 15

# Record noise samples (background/silence)
cargo run --example record_training_samples -- --noise --output-dir cyrup_training -n 3
cargo run --example record_training_samples -- --noise --output-dir cyrup_stop_training -n 3
```

### 2. Train the Models

```bash
# Train using the existing trainer
cargo run --bin koffee train --input cyrup_stop_training --output cyrup_stop.rpw --model-type small

# Or use the example trainer
cargo run --example train_cyrup_stop
```

### 3. Test Your Models

```bash
# Test with both models
cargo run --example test_cyrup_models
```

## Model Configuration

The detectors are configured for optimal wake word detection:

```rust
DetectorConfig {
    avg_threshold: 0.15,      // Lower = more sensitive
    threshold: 0.45,          // Detection threshold
    min_scores: 2,            // Minimum detections needed
    score_mode: ScoreMode::Max,
    vad_mode: Some(VADMode::Easy), // Voice Activity Detection
}
```

## Audio Processing Pipeline

1. **Input**: 16kHz mono audio from microphone
2. **Filters**: 
   - Band-pass filter (85-255 Hz) for human voice
   - Gain normalization for consistent levels
3. **VAD**: Filters out non-speech audio
4. **Detection**: KFC feature extraction + model inference
5. **Output**: Detection events with confidence scores

## Tips for Best Results

1. **Recording Quality**:
   - Quiet environment for wake word samples
   - Natural speech, not too loud or soft
   - Consistent pronunciation across samples
   - Include some background noise samples

2. **Model Training**:
   - At least 10-15 positive samples
   - 2-3 negative (noise) samples
   - Use `ModelType::Small` for faster response
   - Use `ModelType::Medium` for better accuracy

3. **Detection Tuning**:
   - Lower `threshold` for more sensitive detection
   - Increase `min_scores` to reduce false positives
   - Enable VAD to filter out non-speech

## Integration Example

```rust
use koffee::{Kfc, WakewordModel, KoffeeCandleConfig};

// Load models
let wake_model = WakewordModel::load_from_file("syrup.rpw")?;
let stop_model = WakewordModel::load_from_file("cyrup_stop.rpw")?;

// Create detectors
let mut wake_detector = Kfc::new(&config)?;
wake_detector.add_wakeword_model(wake_model)?;

let mut stop_detector = Kfc::new(&config)?;
stop_detector.add_wakeword_model(stop_model)?;

// Process audio
if let Some(detection) = wake_detector.process_bytes(&audio_chunk) {
    if detection.score > 0.5 {
        println!("Wake word detected!");
    }
}
```

## Existing "Syrup" Model

The `syrup.rpw` model was trained on:
- 10 samples of "syrup" (see-rup pronunciation)
- 2 noise samples
- Works well for "cyrup" detection

Training data is in `syrup_training/` directory.

## Troubleshooting

1. **No detections**: Lower the `threshold` value
2. **Too many false positives**: Increase `threshold` and `min_scores`
3. **Slow response**: Use `ModelType::Tiny` or reduce `min_scores`
4. **Background noise issues**: Enable band-pass filter and VAD

## Next Steps

1. Test the existing `syrup.rpw` model with your pronunciation
2. If needed, record custom samples and train new models
3. Tune detection parameters for your environment
4. Integrate into your voice assistant application