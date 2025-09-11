# Fluent Voice Samples

A Rust library and CLI tool for managing and processing voice samples for wake word training.

## Features

- **Parallel Processing**: Uses `jwalk` and `rayon` for efficient directory scanning and processing
- **Audio Metadata Extraction**: Extracts detailed metadata from audio files using `symphonia`
- **YAML Indexing**: Exports and imports sample metadata in YAML format
- **Validation**: Validates audio files for quality and compatibility
- **Progress Reporting**: Provides real-time progress updates during processing

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
fluent-voice-samples = { path = "../fluent-voice-samples" }
```

Or install the CLI tool:

```bash
cargo install --path .
```

## Usage

### As a Library

```rust
use fluent_voice_samples::{index_samples, export_metadata};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Scan a directory for audio files
    let samples = index_samples("path/to/samples")?;
    
    // Export metadata to a YAML file
    export_metadata(&samples, "samples_metadata.yaml")?;
    
    println!("Processed {} samples", samples.len());
    Ok(())
}
```

### CLI Tool

```bash
# Scan a directory and generate an index
fluent-voice-samples scan -i /path/to/samples -o samples.yaml -v

# Export metadata from a directory or existing index
fluent-voice-samples export -i /path/to/samples -o samples.yaml

# Import metadata from a YAML file
fluent-voice-samples import samples.yaml
```

## Directory Structure

```
fluent-voice-samples/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs         # Library entry point
│   ├── error.rs       # Error handling
│   ├── metadata.rs    # Sample metadata
│   ├── progress.rs    # Progress tracking
│   ├── sample.rs      # Sample processing
│   └── scanner.rs     # Directory scanning
└── samples/           # Sample audio files (optional)
```

## License

Licensed under either of:

 * Apache License, Version 2.0
 * MIT license

at your option.
