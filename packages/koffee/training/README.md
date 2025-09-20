# Koffee Training Assets

This directory contains all training data and models for the Koffee wake word detection system.

## 📁 Directory Structure

```
training/
├── wake_words/          # Wake word training samples
│   ├── syrup/          # "syrup" wake word (10 samples + noise)
│   └── sap/            # "sap" wake word (10 samples + noise)
├── unwake_words/       # Unwake word training samples
│   └── cyrup_stop/     # "cyrup stop" unwake (8 samples + noise)
├── noise/              # Consolidated background noise samples
│   ├── noise0.wav
│   └── noise1.wav
└── models/             # Trained production models
    ├── syrup.rpw       # Production "syrup" wake model
    └── cyrup_stop.rpw  # Production "cyrup stop" unwake model
```

## 🎯 Usage Guidelines

### Training Data Requirements
- **Wake Words**: 10-15+ samples recommended
- **Unwake Words**: 8-10+ samples recommended  
- **Noise Samples**: 3-5 samples of background noise
- **Audio Format**: 16kHz mono WAV files
- **Sample Duration**: 2-5 seconds per sample

### File Naming Convention
- Wake word samples: `{word}_{nn}[{label}].wav`
- Unwake word samples: `{phrase}_{nn}[{label}].wav`
- Noise samples: `noise{n}.wav`

### Model Training Commands

```bash
# Train wake word model
cargo run --bin koffee train \
  --data-dir ./training/wake_words/{keyword} \
  --output ./training/models/{keyword}.rpw \
  --model-type small

# Train unwake word model  
cargo run --bin koffee train \
  --data-dir ./training/unwake_words/{stop_phrase} \
  --output ./training/models/{stop_phrase}.rpw \
  --model-type small
```

### Model Testing Commands

```bash
# Real-time detection
cargo run --bin koffee detect \
  --model ./training/models/{keyword}.rpw \
  --stop-model ./training/models/{stop_phrase}.rpw

# Model inspection
cargo run --bin koffee inspect ./training/models/{keyword}.rpw
```

## 📊 Current Models

| Model | Status | Training Data | Quality |
|-------|--------|---------------|---------|
| `syrup.rpw` | ✅ Production | 10 samples + noise | High |
| `cyrup_stop.rpw` | ✅ Production | 8 samples + noise | High |
| `sap.rpw` | ❌ Not trained | 10 samples + noise | - |

## 🔧 Advanced Configuration

### Model Types
- **Tiny**: Fastest inference, lower accuracy
- **Small**: Balanced speed/accuracy (recommended)
- **Medium**: Higher accuracy, slower inference
- **Large**: Highest accuracy, slowest inference

### Training Parameters
- **Learning Rate**: 0.0005-0.002 (lower = more stable)
- **Epochs**: 20-100 (more = better convergence)
- **Batch Size**: 2-16 (smaller = more stable for small datasets)

## 🚀 Best Practices

1. **Record in Quiet Environment**: Minimize background noise during recording
2. **Natural Pronunciation**: Use consistent, natural speech patterns
3. **Diverse Samples**: Include slight variations in tone and speed
4. **Adequate Noise Data**: Include representative background noise
5. **Test Thoroughly**: Validate models in target deployment environment