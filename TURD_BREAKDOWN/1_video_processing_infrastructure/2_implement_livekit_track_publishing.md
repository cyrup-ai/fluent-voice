# Task: Remove LiveKit Track Publishing Stub

**Priority**: 🚨 CRITICAL (Code Quality Violation)  
**File**: [`packages/video/src/main.rs:218-220`](../../packages/video/src/main.rs#L218)  
**Milestone**: 1_video_processing_infrastructure  
**Status**: ✅ **VERIFIED CURRENT** - Codebase analysis confirms all details accurate

## Problem Description

Explicit stubbing violates CLAUDE.md code quality standards:

```rust
// CURRENT (EXPLICIT STUB VIOLATION - VERIFIED):
if let Some(_track) = &local_track {
    // This is a placeholder - actual implementation would convert VideoTrack to livekit::track::LocalTrack
    // room_clone.local_participant().publish_track(...).await.ok();
}
```

**Impact**: 
- Violates "no stubbing" rule in CLAUDE.md conventions
- Non-production quality code with placeholder comments
- Missing core video streaming functionality
- Code review failure due to explicit stub violation

## Critical Discovery: Implementation Pattern Already Exists! ⚡

**MAJOR FINDING**: The LiveKit integration framework is substantially complete! The stubbing only blocks the final video track publishing step.

### Existing Working Infrastructure

**1. LiveKit Audio Track Publishing Pattern - Working** [`packages/livekit/src/livekit_client.rs:84-100`](../../packages/livekit/src/livekit_client.rs#L84):
```rust
pub async fn publish_local_microphone_track(&self) -> Result<(LocalTrackPublication, AudioStream)> {
    let (track, stream) = self.playback.capture_local_microphone_track()?;
    let publication = self
        .local_participant()
        .publish_track(
            livekit::track::LocalTrack::Audio(track.0),  // ← EXACT PATTERN for video
            livekit::options::TrackPublishOptions {
                source: livekit::track::TrackSource::Microphone,  // ← Use Camera/ScreenShare
                ..Default::default()
            },
        )
        .await?;
    Ok((publication, stream))
}
```

**2. VideoTrack Stream API - Complete** [`packages/video/src/track.rs:112-133`](../../packages/video/src/track.rs#L112):
```rust
pub fn get_frame_stream(&self) -> impl Stream<Item = VideoFrame> + Send + 'static {
    futures::stream::unfold(current_frame, |current_frame| async move {
        let frame = if let Ok(read_guard) = current_frame.read() {
            read_guard.clone()
        } else { None };
        
        tokio::time::sleep(Duration::from_millis(16)).await; // ~60fps
        
        if let Some(frame) = frame {
            Some((frame, current_frame))
        } else {
            Some((VideoFrame::default(), current_frame))
        }
    })
    .filter(|frame| futures::future::ready(!frame.is_empty()))
}
```

**3. LiveKit Room Integration - Available** [`packages/video/src/main.rs:187-189`](../../packages/video/src/main.rs#L187):
```rust
let config = livekit::RoomOptions::default();
let (_room, mut events) = livekit::Room::connect(url, token, config).await?;
// Room connection and local_participant() already working
```

### What's Actually Missing (Small Implementation Gap)

**ONLY Missing**: Replace stub with fluent-voice VideoTrack → LiveKit LocalVideoTrack conversion

**Current Issue** [`packages/video/src/main.rs:218-220`](../../packages/video/src/main.rs#L218):
```rust
// This is a placeholder - actual implementation would convert VideoTrack to livekit::track::LocalTrack
// room_clone.local_participant().publish_track(...).await.ok();
```

## Research-Driven Implementation Plan

### LiveKit Video Track Creation Pattern

**Reference Implementation**: [`./tmp/livekit-rust-sdks/examples/wgpu_room/src/logo_track.rs:63-95`](../../tmp/livekit-rust-sdks/examples/wgpu_room/src/logo_track.rs#L63)

```rust
// COMPLETE WORKING PATTERN:
pub async fn publish(&mut self) -> Result<(), RoomError> {
    // 1. Create NativeVideoSource with resolution
    let rtc_source = NativeVideoSource::new(VideoResolution {
        width: 1920,
        height: 1080,
    });

    // 2. Create LocalVideoTrack from RtcVideoSource
    let track = LocalVideoTrack::create_video_track(
        "fluent_video_track",
        RtcVideoSource::Native(rtc_source.clone()),
    );

    // 3. Publish track to room with appropriate source type
    self.room
        .local_participant()
        .publish_track(
            LocalTrack::Video(track.clone()),
            TrackPublishOptions {
                source: TrackSource::Camera,  // or ScreenShare
                simulcast: false,
                video_codec: VideoCodec::H264,
                ..Default::default()
            },
        )
        .await?;
        
    Ok(())
}
```

### Frame Feeding Pattern

**Reference Implementation**: [`./tmp/livekit-rust-sdks/examples/wgpu_room/src/logo_track.rs:156-199`](../../tmp/livekit-rust-sdks/examples/wgpu_room/src/logo_track.rs#L156)

```rust
// Frame feeding loop pattern:
async fn track_task(rtc_source: NativeVideoSource, frame_stream: FrameStream) {
    let mut interval = tokio::time::interval(Duration::from_millis(1000 / 30)); // 30fps
    
    // Create I420 buffer for LiveKit format
    let mut video_frame = VideoFrame {
        rotation: VideoRotation::VideoRotation0,
        buffer: I420Buffer::new(1920, 1080),
        timestamp_us: 0,
    };
    
    loop {
        interval.tick().await;
        
        // Get next frame from fluent-voice VideoTrack
        if let Some(fluent_frame) = frame_stream.next().await {
            // Convert fluent-voice RGBA to LiveKit I420
            convert_rgba_to_i420(&fluent_frame, &mut video_frame.buffer);
            
            // Update timestamp
            video_frame.timestamp_us = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as i64;
            
            // Feed frame into LiveKit source
            rtc_source.capture_frame(&video_frame);
        }
    }
}
```

### RGBA to I420 Conversion

**Reference Pattern**: [`./tmp/livekit-rust-sdks/examples/wgpu_room/src/logo_track.rs:182-193`](../../tmp/livekit-rust-sdks/examples/wgpu_room/src/logo_track.rs#L182)

```rust
use livekit::webrtc::native::yuv_helper;

fn convert_rgba_to_i420(
    rgba_frame: &crate::VideoFrame,  // fluent-voice VideoFrame
    i420_buffer: &mut I420Buffer     // LiveKit I420Buffer
) -> Result<()> {
    let rgba_data = rgba_frame.to_rgba_bytes()?;
    let width = rgba_frame.width() as i32;
    let height = rgba_frame.height() as i32;
    
    let (stride_y, stride_u, stride_v) = i420_buffer.strides();
    let (data_y, data_u, data_v) = i420_buffer.data_mut();
    
    // Convert RGBA to I420 using LiveKit's yuv_helper
    yuv_helper::abgr_to_i420(
        &rgba_data,
        (width * 4) as u32,  // RGBA stride
        data_y,
        stride_y,
        data_u, 
        stride_u,
        data_v,
        stride_v,
        width,
        height,
    );
    
    Ok(())
}
```

### macOS CVPixelBuffer Optimization

**Reference Implementation**: [`./tmp/livekit-rust-sdks/libwebrtc/src/video_frame.rs:459-461`](../../tmp/livekit-rust-sdks/libwebrtc/src/video_frame.rs#L459)

```rust
// For macOS efficiency - direct CVPixelBuffer integration
#[cfg(target_os = "macos")]
fn create_native_buffer_from_cv_pixel_buffer(
    cv_pixel_buffer: CVPixelBuffer
) -> Result<NativeBuffer> {
    let cv_ptr = cv_pixel_buffer.as_ptr() as *mut std::ffi::c_void;
    
    // Create LiveKit NativeBuffer directly from CVPixelBuffer
    let native_buffer = unsafe {
        NativeBuffer::from_cv_pixel_buffer(cv_ptr)
    };
    
    Ok(native_buffer)
}

// Use NativeBuffer in VideoFrame for zero-copy efficiency
let video_frame = VideoFrame {
    rotation: VideoRotation::VideoRotation0,
    buffer: Box::new(native_buffer),
    timestamp_us: frame_timestamp,
};
```

## Revised Implementation Steps (Infrastructure-Aware)

### 1. **Create VideoTrack to LocalVideoTrack Conversion** (2 hours)
**File**: [`packages/video/src/main.rs:218-220`](../../packages/video/src/main.rs#L218)
```rust
// REPLACE stubbed lines 218-220 with:
if let Some(video_track) = &local_track {
    let local_video_track = create_local_video_track_from_fluent_track(video_track)?;
    let options = TrackPublishOptions {
        source: TrackSource::Camera,  // or ScreenShare based on source
        simulcast: false,
        video_codec: VideoCodec::H264,
        ..Default::default()
    };
    
    let _publication = room_clone
        .local_participant()
        .publish_track(
            LocalTrack::Video(local_video_track),
            options
        )
        .await?;
    
    info!("Video track published successfully");
}
```

### 2. **Implement Conversion Function** (2.5 hours)
```rust
fn create_local_video_track_from_fluent_track(
    fluent_track: &crate::VideoTrack
) -> Result<livekit::track::LocalVideoTrack> {
    use livekit::webrtc::video_source::{RtcVideoSource, VideoResolution, native::NativeVideoSource};
    use livekit::webrtc::video_frame::{I420Buffer, VideoFrame, VideoRotation};
    use tokio::sync::oneshot;
    
    // Create video source matching track dimensions
    let resolution = VideoResolution {
        width: fluent_track.width(),
        height: fluent_track.height(),
    };
    let rtc_source = NativeVideoSource::new(resolution);
    
    // Create LiveKit track
    let local_track = LocalVideoTrack::create_video_track(
        "fluent_video_track",
        RtcVideoSource::Native(rtc_source.clone()),
    );
    
    // Start frame feeding task
    let frame_stream = fluent_track.get_frame_stream();
    tokio::spawn(async move {
        feed_frames_to_source(rtc_source, frame_stream).await;
    });
    
    Ok(local_track)
}
```

### 3. **Implement Frame Feeding Task** (2 hours)
```rust
async fn feed_frames_to_source(
    rtc_source: NativeVideoSource,
    mut frame_stream: impl Stream<Item = crate::VideoFrame> + Send + 'static
) {
    use futures::StreamExt;
    use livekit::webrtc::video_frame::{I420Buffer, VideoFrame, VideoRotation};
    
    let mut interval = tokio::time::interval(Duration::from_millis(33)); // ~30fps
    
    while let Some(fluent_frame) = frame_stream.next().await {
        interval.tick().await;
        
        // Create I420 buffer for LiveKit
        let mut i420_buffer = I420Buffer::new(fluent_frame.width(), fluent_frame.height());
        
        // Convert fluent-voice frame to LiveKit format
        if let Err(e) = convert_fluent_frame_to_i420(&fluent_frame, &mut i420_buffer) {
            error!("Frame conversion failed: {}", e);
            continue;
        }
        
        // Create LiveKit VideoFrame
        let video_frame = VideoFrame {
            rotation: VideoRotation::VideoRotation0,
            buffer: i420_buffer,
            timestamp_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as i64,
        };
        
        // Feed frame to LiveKit source
        rtc_source.capture_frame(&video_frame);
    }
}
```

### 4. **Add Format Conversion Implementation** (1.5 hours)
```rust
use livekit::webrtc::native::yuv_helper;

fn convert_fluent_frame_to_i420(
    fluent_frame: &crate::VideoFrame,
    i420_buffer: &mut I420Buffer
) -> Result<()> {
    // Get RGBA data from fluent-voice frame
    let rgba_data = fluent_frame.to_rgba_bytes()
        .map_err(|e| anyhow::anyhow!("Failed to get RGBA data: {}", e))?;
    
    let width = fluent_frame.width() as i32;
    let height = fluent_frame.height() as i32;
    
    // Get I420 buffer pointers
    let (stride_y, stride_u, stride_v) = i420_buffer.strides();
    let (data_y, data_u, data_v) = i420_buffer.data_mut();
    
    // Convert RGBA to I420 using LiveKit's optimized converter
    yuv_helper::abgr_to_i420(
        &rgba_data,
        (width * 4) as u32,  // RGBA stride (4 bytes per pixel)
        data_y,
        stride_y,
        data_u,
        stride_u,
        data_v,
        stride_v,
        width,
        height,
    );
    
    Ok(())
}
```

### 5. **Add Comprehensive Error Handling** (1 hour)
```rust
#[derive(Debug, thiserror::Error)]
pub enum VideoTrackConversionError {
    #[error("VideoTrack stream ended unexpectedly")]
    StreamEnded,
    
    #[error("Frame conversion failed: {source}")]
    ConversionFailed { source: anyhow::Error },
    
    #[error("LiveKit track creation failed: {reason}")]
    TrackCreationFailed { reason: String },
    
    #[error("Unsupported video resolution: {width}x{height}")]
    UnsupportedResolution { width: u32, height: u32 },
    
    #[error("Frame format conversion failed: {details}")]
    FormatConversionFailed { details: String },
}
```

### 6. **Add LiveKit Dependencies** (15 minutes)
**File**: [`packages/video/Cargo.toml`](../../packages/video/Cargo.toml)
```toml
[dependencies]
# LiveKit integration (already exists in livekit package)
livekit = { path = "../livekit" }
# Video processing
futures = "0.3"
tokio = { version = "1.0", features = ["time"] }
```

### 7. **Remove Dead Code Suppressions** (30 minutes)
**File**: [`packages/video/src/main.rs:176, 235`](../../packages/video/src/main.rs#L176)
- Remove `#[allow(dead_code)]` annotations at lines 176 and 235
- Implement or remove any dead code that surfaces
- Ensure clean compilation with `cargo check --message-format short --quiet`

### 8. **Testing and Validation** (1.5 hours)
```rust
#[tokio::test]
async fn test_video_track_conversion() {
    let video_track = create_test_video_track();
    let local_track = create_local_video_track_from_fluent_track(&video_track).unwrap();
    
    // Verify LocalVideoTrack is created successfully
    assert_eq!(local_track.name(), "fluent_video_track");
    assert_eq!(local_track.kind(), TrackKind::Video);
}

#[tokio::test]
async fn test_track_publishing() {
    let room = connect_test_room().await.unwrap();
    let video_track = create_test_video_track();
    
    let local_track = create_local_video_track_from_fluent_track(&video_track).unwrap();
    let result = room.local_participant().publish_track(
        LocalTrack::Video(local_track),
        TrackPublishOptions {
            source: TrackSource::Camera,
            ..Default::default()
        }
    ).await;
    
    assert!(result.is_ok());
}
```

## Available Research Resources

### LiveKit SDK Reference Implementation
**Primary Reference**: [`./tmp/livekit-rust-sdks/examples/wgpu_room/src/logo_track.rs`](../../tmp/livekit-rust-sdks/examples/wgpu_room/src/logo_track.rs)  
- Complete video track creation and publishing workflow
- Frame feeding loop with I420 conversion
- Track lifecycle management (publish/unpublish)
- Performance-optimized frame timing

### LiveKit Core APIs
**Reference**: [`./tmp/livekit-rust-sdks/livekit/src/room/track/local_video_track.rs:40-65`](../../tmp/livekit-rust-sdks/livekit/src/room/track/local_video_track.rs#L40)
- `LocalVideoTrack::create_video_track()` API signature
- `LocalVideoTrack::new()` constructor pattern
- Integration with RtcVideoSource types

### Video Source Implementation
**Reference**: [`./tmp/livekit-rust-sdks/libwebrtc/src/video_source.rs:65-77`](../../tmp/livekit-rust-sdks/libwebrtc/src/video_source.rs#L65)
- `NativeVideoSource::new(VideoResolution)` constructor
- `capture_frame<T: AsRef<dyn VideoBuffer>>(&self, frame: &VideoFrame<T>)` method
- VideoResolution structure definition

### CVPixelBuffer Integration
**Reference**: [`./tmp/livekit-rust-sdks/libwebrtc/src/video_frame.rs:459-471`](../../tmp/livekit-rust-sdks/libwebrtc/src/video_frame.rs#L459)
- `NativeBuffer::from_cv_pixel_buffer()` for macOS efficiency
- Zero-copy video frame integration patterns
- CVPixelBuffer safety requirements

### Existing Audio Track Pattern
**Reference**: [`packages/livekit/src/livekit_client.rs:84-100`](../../packages/livekit/src/livekit_client.rs#L84)
- Complete working publish_track() pattern
- TrackPublishOptions configuration
- LocalTrackPublication handling
- Error handling and async patterns

## Success Criteria

- [ ] **No Stubbing Violations**: All placeholder comments and explicit stubs removed
- [ ] **Real Video Publishing**: fluent-voice VideoTrack frames stream to LiveKit room successfully  
- [ ] **Format Conversion**: RGBA to I420 conversion works correctly with minimal quality loss
- [ ] **Performance**: 30fps real-time processing with <33ms latency per frame
- [ ] **Error Handling**: Meaningful error messages for conversion and publishing failures
- [ ] **Cross-Platform Ready**: macOS CVPixelBuffer optimization with generic fallback
- [ ] **Integration**: Works seamlessly with existing Room connection and local_participant APIs

## Platform Support Matrix

| Platform | RGBA→I420 Conversion | CVPixelBuffer Optimization | Zero-Copy Support |
|----------|---------------------|----------------------------|-------------------|
| macOS | ✅ yuv_helper | ✅ NativeBuffer::from_cv_pixel_buffer | ✅ Direct integration |
| Linux | ✅ yuv_helper | ❌ Generic buffer only | ⚠️ Limited |
| Windows | ✅ yuv_helper | ❌ Generic buffer only | ⚠️ Limited |

## Risk Assessment

**Risk Level**: CRITICAL - Violates core code quality standards  
**Revised Effort Estimate**: 8-10 hours (optimized due to existing infrastructure)  
**Complexity**: Medium-High (async stream integration + video format conversion)

**Risks**:
- Frame format mismatch between fluent-voice RGBA and LiveKit I420 requirements
- Performance bottlenecks in real-time RGBA→I420 conversion pipeline  
- Async lifetime management in stream conversion and task coordination
- Memory pressure from video frame buffer allocation

**Mitigations**:
- Use proven LiveKit yuv_helper for optimized RGBA→I420 conversion
- Follow existing audio track pattern for publish_track() implementation
- Profile conversion performance with real video streams during development
- Implement CVPixelBuffer optimization for macOS zero-copy efficiency
- Test with multiple video sources (camera, screen) to ensure robustness

## Completion Definition

Task is complete when:
1. ✅ No stubbing violations remain in connect_livekit function (lines 218-220 fixed)
2. ✅ `cargo check --message-format short --quiet` passes without warnings
3. ✅ No `#[allow(dead_code)]` suppressions remain (lines 176, 235 cleaned)
4. ✅ fluent-voice VideoTrack successfully publishes to LiveKit room
5. ✅ Video frames appear correctly in remote LiveKit clients  
6. ✅ Performance meets real-time video streaming requirements (30fps)
7. ✅ Format conversion works correctly (RGBA→I420 with <5% quality loss)
8. ✅ Error handling provides meaningful feedback for conversion and publishing failures
9. ✅ Integration tests pass demonstrating full VideoTrack → LiveKit streaming pipeline

## Dependencies Resolution

**Before Starting**: LiveKit Rust SDK cloned to ./tmp for reference patterns  
**Parallel Work**: Can develop alongside screen capture and video frame extraction tasks  
**Completion Blockers**: Must have working VideoTrack instance with real frames to test against

## Implementation Priority

This task removes a critical code quality violation while enabling core video streaming functionality. With the audio track publishing pattern as a proven reference and the LiveKit SDK providing complete integration examples, this becomes a **focused 8-10 hour implementation** to bridge fluent-voice VideoTrack to LiveKit LocalVideoTrack.

**Key Insight**: The hard work (Room connection, participant management, track APIs) is already done. We just need to replace stub comments with `LocalVideoTrack::create_video_track()` + frame conversion calls.