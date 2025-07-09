# Reqwest HTTP/3 QUIC Upgrade Guide

Based on research of reqwest 0.12.22 source code and examples.

## Current Status

HTTP/3 support in reqwest is **experimental and unstable**. It requires:

1. **Feature flag**: `http3`
2. **Unstable compiler flag**: `RUSTFLAGS="--cfg reqwest_unstable"`
3. **Dependencies**: Uses `h3`, `h3-quinn`, `quinn` for QUIC implementation

## Cargo.toml Configuration

```toml
[dependencies]
reqwest = { version = "0.12", features = ["http3", "rustls-tls-manual-roots"] }
tokio = { version = "1", features = ["full"] }
```

Note: HTTP/3 requires `rustls-tls-manual-roots`, not native-tls.

## Build Configuration

### Environment Variable
```bash  
export RUSTFLAGS="--cfg reqwest_unstable"
```

### Or in .cargo/config.toml
```toml
[build]
rustflags = ["--cfg", "reqwest_unstable"]
```

## Client Configuration

### Basic HTTP/3 Client
```rust
let client = reqwest::Client::builder()
    .http3_prior_knowledge()  // Force HTTP/3
    .build()?;

let response = client
    .get("https://example.com")
    .version(http::Version::HTTP_3)
    .send()
    .await?;
```

### Advanced QUIC Tuning
```rust
let client = reqwest::Client::builder()
    .http3_prior_knowledge()
    .tls_early_data(true)                        // Enable 0-RTT
    .http3_max_idle_timeout(Duration::from_secs(30))
    .http3_stream_receive_window(1024 * 1024)    // 1MB per stream
    .http3_conn_receive_window(10 * 1024 * 1024)  // 10MB total
    .http3_send_window(1024 * 1024)              // 1MB send
    .build()?;
```

## API Methods

### Client Builder Methods
- `http3_prior_knowledge()` - Force HTTP/3 without HTTP/1.1 fallback
- `tls_early_data(bool)` - Enable/disable 0-RTT data 
- `http3_max_idle_timeout(Duration)` - QUIC connection timeout
- `http3_stream_receive_window(u64)` - Per-stream receive buffer
- `http3_conn_receive_window(u64)` - Connection-wide receive buffer  
- `http3_send_window(u64)` - Send buffer size

### Request Configuration
```rust
request.version(http::Version::HTTP_3)
```

## Implementation Strategy for fluent-voice

### Phase 1: Feature Flag Support
1. Add `http3` feature to fluent-voice crates
2. Conditional compilation with `#[cfg(feature = "http3")]`
3. Fallback to HTTP/2 when HTTP/3 unavailable

### Phase 2: ElevenLabs Client Upgrade
Replace the current multipart-heavy HTTP client:

```rust
#[cfg(feature = "http3")]
pub fn new_http3_client() -> Result<reqwest::Client, Error> {
    Ok(reqwest::Client::builder()
        .http3_prior_knowledge()
        .tls_early_data(true)
        .http3_max_idle_timeout(Duration::from_secs(30))
        .build()?)
}

#[cfg(not(feature = "http3"))]
pub fn new_http3_client() -> Result<reqwest::Client, Error> {
    // Fallback to HTTP/2
    Ok(reqwest::Client::new())
}
```

### Phase 3: fluent-voice Engine Integration
Expose HTTP/3 configuration through the engine interface:

```rust
pub struct HttpConfig {
    pub enable_http3: bool,
    pub enable_early_data: bool,
    pub max_idle_timeout: Duration,
    pub stream_receive_window: u64,
    pub conn_receive_window: u64,
}

impl VoiceEngine {
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = config;
        self
    }
}
```

## Benefits of HTTP/3/QUIC

1. **Reduced Latency**: 0-RTT connection establishment
2. **Better Reliability**: No head-of-line blocking
3. **Improved Performance**: Multiplexed streams over UDP
4. **Mobile Optimized**: Connection migration support
5. **Future-Proof**: Latest HTTP standard

## Considerations

1. **Experimental**: Still unstable in reqwest
2. **Server Support**: Target APIs must support HTTP/3
3. **Network**: Some networks block UDP traffic
4. **Debugging**: Fewer tools for HTTP/3 debugging
5. **Dependencies**: Adds quinn, h3 dependency weight

## Testing Strategy

1. Feature-gated unit tests
2. Integration tests with HTTP/3 test servers
3. Fallback testing when HTTP/3 unavailable
4. Performance benchmarks vs HTTP/2

## Next Steps

1. Check ElevenLabs API HTTP/3 support
2. Implement feature-gated HTTP/3 client  
3. Add configuration to fluent-voice engine
4. Performance testing and optimization