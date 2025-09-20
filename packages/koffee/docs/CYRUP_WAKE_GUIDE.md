# Cyrup Wake Word System Guide

This guide explains how to use the "cyrup" wake word and "cyrup stop" unwake command system in Koffee.

## 📁 Training Assets Organization

All training assets are now organized in the `./training/` directory:

```
training/
├── wake_words/          # Wake word training samples
│   ├── syrup/          # "syrup" wake word samples (10 samples)
│   └── sap/            # "sap" wake word samples (10 samples)
├── unwake_words/       # Unwake word training samples  
│   └── cyrup_stop/     # "cyrup stop" unwake samples (8 samples)
├── noise/              # Background noise samples
│   ├── noise0.wav
│   └── noise1.wav
└── models/             # Trained production models
    ├── syrup.rpw       # Production "syrup" wake model
    └── cyrup_stop.rpw  # Production "cyrup stop" unwake model
```

## Quick Start (Using Existing Models)

The repository includes trained models for "syrup" (pronounced "see-rup") and "cyrup stop":

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

## 🎓 Training Custom Wake Words

### Step 1: Record Training Samples

Record samples for your custom wake word using the CLI:

```bash
# Record wake word samples (15+ recommended for accuracy)
cargo run --bin koffee record \
  --output-dir ./training/wake_words/my_keyword \
  --label "my_keyword" \
  --count 15 \
  --duration 3

# Record unwake word samples (10+ recommended)
cargo run --bin koffee record \
  --output-dir ./training/unwake_words/stop_keyword \
  --label "stop_keyword" \
  --count 10 \
  --duration 3

# Record noise samples (3-5 recommended)
cargo run --bin koffee record \
  --output-dir ./training/noise \
  --label "noise" \
  --count 5 \
  --duration 5
```

### Step 2: Generate Synthetic Samples (Advanced)

Enhance training data with TTS-generated samples:

```bash
# Generate additional synthetic samples
cargo run --bin koffee generate \
  --phrase "my keyword" \
  --output-dir ./training/wake_words/my_keyword_synthetic \
  --count 20 \
  --timber warm \
  --speed normal \
  --noise-reduction true
```

### Step 3: Train Your Models

```bash
# Train wake word model
cargo run --bin koffee train \
  --data-dir ./training/wake_words/my_keyword \
  --output ./training/models/my_keyword.rpw \
  --model-type small \
  --epochs 50 \
  --learning-rate 0.001

# Train unwake word model (optional)
cargo run --bin koffee train \
  --data-dir ./training/unwake_words/stop_keyword \
  --output ./training/models/stop_keyword.rpw \
  --model-type small \
  --epochs 50
```

### Step 4: Test Your Models

```bash
# Test real-time detection
cargo run --bin koffee detect \
  --model ./training/models/my_keyword.rpw \
  --stop-model ./training/models/stop_keyword.rpw \
  --threshold 0.5

# Inspect model details
cargo run --bin koffee inspect ./training/models/my_keyword.rpw

# List available audio devices
cargo run --bin koffee list-devices
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

## 📊 Existing Models

### Syrup Wake Model (`./training/models/syrup.rpw`)
- **Training Data**: 10 samples of "syrup" (see-rup pronunciation) + 2 noise samples
- **Location**: `./training/wake_words/syrup/`
- **Usage**: Works well for "cyrup" detection
- **Quality**: Production-ready

### Cyrup Stop Unwake Model (`./training/models/cyrup_stop.rpw`)
- **Training Data**: 8 samples of "cyrup stop" + 2 noise samples  
- **Location**: `./training/unwake_words/cyrup_stop/`
- **Usage**: Deactivates wake word detection
- **Quality**: Production-ready

### Sap Wake Model (Training Data Available)
- **Training Data**: 10 samples of "sap" + 2 noise samples
- **Location**: `./training/wake_words/sap/`
- **Status**: Training data available, model not yet trained

## Troubleshooting

1. **No detections**: Lower the `threshold` value
2. **Too many false positives**: Increase `threshold` and `min_scores`
3. **Slow response**: Use `ModelType::Tiny` or reduce `min_scores`
4. **Background noise issues**: Enable band-pass filter and VAD

## 🚀 Advanced Training Configuration

### High-Accuracy Training (Slower)
```bash
cargo run --bin koffee train \
  --data-dir ./training/wake_words/my_keyword \
  --output ./training/models/my_keyword_accurate.rpw \
  --model-type medium \
  --epochs 100 \
  --learning-rate 0.0005 \
  --batch-size 4
```

### Fast Training (Lower Accuracy)
```bash
cargo run --bin koffee train \
  --data-dir ./training/wake_words/my_keyword \
  --output ./training/models/my_keyword_fast.rpw \
  --model-type tiny \
  --epochs 20 \
  --learning-rate 0.002 \
  --batch-size 16
```

## 🎯 Next Steps

1. **Test Existing Models**: Use `syrup.rpw` and `cyrup_stop.rpw` with your pronunciation
2. **Train Custom Models**: Follow the training workflow for your specific wake words
3. **Optimize Parameters**: Tune detection thresholds for your environment
4. **Production Integration**: Integrate into your voice assistant applications
5. **Scale Training Data**: Add more samples for improved accuracy