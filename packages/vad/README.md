# Speakrs Turn Detector

A Rust library for voice activity detection and speech turn detection in audio streams. This crate provides tools to analyze audio in real-time or from recorded sources to detect when someone is speaking and when turns in conversation occur.

## Features

- Voice Activity Detection (VAD) using the Silero VAD model
- Real-time processing of audio streams
- Support for both synchronous and asynchronous processing
- Customizable detection parameters
- Lightweight and efficient implementation
- Supports iterator-based and stream-based processing

## Installation

Add this dependency to your Cargo.toml:

```toml
[dependencies]
speakrs-turndetector = { version = "0.1.0" }
```

To enable async stream support, add the `async` feature:

```toml
[dependencies]
speakrs-turndetector = { version = "0.1.0", features = ["async"] }
```

## Usage

### Basic Voice Activity Detection

```rust
use speakrs_turndetector::{VoiceActivityDetector, Sample};

// Create a VAD detector with appropriate parameters for your audio
let mut detector = VoiceActivityDetector::builder()
    .chunk_size(1536)  // 96ms at 16kHz
    .sample_rate(16000)
    .build()
    .expect("Failed to create VAD detector");

// Process audio samples and get speech probability
let audio_samples: Vec<f32> = vec![/* your audio samples here */];
let speech_probability = detector.predict(audio_samples);

println!("Speech probability: {}", speech_probability);
```

### Processing Audio Iteratively

```rust
use speakrs_turndetector::{IteratorExt, Sample};

fn process_audio<S: Sample, I: Iterator<Item = S>>(audio: I) {
    let predictions = audio
        .chunks(1536)
        .predict_speech(16000, 0.5);
    
    for is_speech in predictions {
        println!("Speech detected: {}", is_speech);
    }
}
```

## License

This crate is licensed under the terms of the MIT license.