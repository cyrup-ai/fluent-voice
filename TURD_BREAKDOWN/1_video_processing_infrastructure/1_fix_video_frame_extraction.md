# Task: Fix Video Frame Extraction from Black Placeholders

**Priority**: 🔇 MEDIUM (Silent Failure)  
**File**: [`packages/livekit/src/playback.rs:746-754`](../../packages/livekit/src/playback.rs#L746)  
**Milestone**: 1_video_processing_infrastructure  

## Problem Description

Video frame extraction returns black placeholder data instead of real frames:

```rust
// CURRENT (SILENT FAILURE) - packages/livekit/src/playback.rs:746-754:
unsafe fn get_buffer_data(
    _frame: &RemoteVideoFrame,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // This is a placeholder implementation - needs proper frame buffer access
    let _width = 1920usize; // Default width
    let _height = 1080usize; // Default height
    let bytes_per_row = _width * 4; // Assuming RGBA format
    let buffer_size = bytes_per_row * _height;
    let data = vec![0u8; buffer_size]; // Black frame as placeholder
    Ok(data)
}
```

**Impact**: 
- Video functionality appears to work but shows only black frames
- Silent failure makes debugging extremely difficult
- Users see broken video without clear error indication
- RemoteVideoFrame data is completely ignored despite containing real video data

## Critical Discovery: Infrastructure Already Exists! ⚡

**MAJOR FINDING**: This is NOT a complex build-from-scratch task! The video frame infrastructure is substantially complete:

### Existing Working Infrastructure

**1. VideoFrameExtensions Trait - Already Implemented** [`packages/livekit/src/playback.rs:667-710`](../../packages/livekit/src/playback.rs#L667):
```rust
pub trait VideoFrameExtensions {
    fn to_rgba_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>>;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
}

impl VideoFrameExtensions for RemoteVideoFrame {
    fn to_rgba_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        // Uses get_buffer_data() - the function that needs fixing
        let frame_data = unsafe { get_buffer_data(self)? };
        // ... BGRA to RGBA conversion logic already implemented
    }
}
```

**2. VideoFrame API - Complete Infrastructure** [`packages/video/src/video_frame.rs:35-45`](../../packages/video/src/video_frame.rs#L35):
```rust
impl VideoFrame {
    pub fn to_rgba_bytes(&self) -> Result<Vec<u8>> {
        self.inner.to_rgba_bytes()
    }
    pub fn width(&self) -> u32 { self.inner.width() }
    pub fn height(&self) -> u32 { self.inner.height() }
    pub fn cv_buffer(&self) -> Option<&core_video::image_buffer::CVImageBuffer>
}
```

**3. Thread-Safe Core Video Integration - Working** [`packages/video/src/macos.rs:21-51`](../../packages/video/src/macos.rs#L21):
```rust
#[derive(Clone)]
pub struct ThreadSafeCVImageBuffer {
    inner: CVImageBuffer,
}

// SAFETY: CVImageBuffer is thread-safe for sharing across threads when properly
// retained/released. Core Video's reference counting ensures memory safety.
unsafe impl Send for ThreadSafeCVImageBuffer {}
unsafe impl Sync for ThreadSafeCVImageBuffer {}
```

**4. CVPixelBuffer Type Definition - Available** [`packages/livekit/src/playback.rs:825`](../../packages/livekit/src/playback.rs#L825):
```rust
#[cfg(target_os = "macos")]
pub type RemoteVideoFrame = core_video::pixel_buffer::CVPixelBuffer;
```

### What's Actually Missing (Small Implementation Gap)

**ONLY Missing**: Replace black placeholder with real CVPixelBuffer data extraction

**Current Issue** [`packages/livekit/src/playback.rs:746-754`](../../packages/livekit/src/playback.rs#L746):
```rust
// get_buffer_data() - BROKEN but everything else works
let data = vec![0u8; buffer_size]; // Creates black frame instead of reading CVPixelBuffer
```

## Research-Driven Implementation Plan

### Implementation Pattern - Core Graphics Data Extraction

**Reference Implementation**: [`./tmp/screenshots-rs/src/macos/capture.rs:9-49`](../../tmp/screenshots-rs/src/macos/capture.rs#L9)

```rust
use objc2_core_graphics::{
    CGDataProvider, CGImage, CGWindowImageOption, CGWindowListCreateImage,
    CGWindowListOption,
};

pub fn extract_rgba_from_cgimage(cg_image: &CGImage) -> Result<Vec<u8>> {
    unsafe {
        let width = CGImage::width(cg_image.as_deref());
        let height = CGImage::height(cg_image.as_deref());
        let data_provider = CGImage::data_provider(cg_image.as_deref());

        let data = CGDataProvider::data(data_provider.as_deref())
            .ok_or_else(|| anyhow::anyhow!("Failed to copy screen capture data"))?
            .to_vec();

        let bytes_per_row = CGImage::bytes_per_row(cg_image.as_deref());

        // Handle row padding - macOS can have extra bytes at row end
        let mut rgba_buffer = Vec::with_capacity(width * height * 4);
        for row in data.chunks_exact(bytes_per_row) {
            rgba_buffer.extend_from_slice(&row[..width * 4]);
        }

        // BGRA -> RGBA conversion (Core Graphics uses BGRA)
        for bgra in rgba_buffer.chunks_exact_mut(4) {
            bgra.swap(0, 2); // B <-> R
        }

        Ok(rgba_buffer)
    }
}
```

### CVPixelBuffer Integration Pattern 

**Based on Core Video API Documentation and existing MacOSVideoFrame patterns**:

```rust
// REPLACEMENT for get_buffer_data() function
unsafe fn get_buffer_data(
    frame: &RemoteVideoFrame,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use core_video::pixel_buffer::CVPixelBufferLockFlags;
    
    // Get real dimensions from CVPixelBuffer
    let width = frame.width() as usize;
    let height = frame.height() as usize;
    
    // Detect pixel format for proper conversion
    let pixel_format = frame.pixel_format_type();
    
    // Lock the pixel buffer for reading (critical for memory safety)
    let lock_result = frame.lock_base_address(CVPixelBufferLockFlags::ReadOnly);
    if lock_result != 0 {
        return Err(format!("Failed to lock pixel buffer: {}", lock_result).into());
    }
    
    // Ensure buffer is unlocked on scope exit
    let _unlock_guard = defer(|| {
        frame.unlock_base_address(CVPixelBufferLockFlags::ReadOnly);
    });
    
    // Get raw pixel data pointer and stride information
    let base_address = frame.base_address();
    let bytes_per_row = frame.bytes_per_row();
    let buffer_size = bytes_per_row * height;
    
    if base_address.is_null() {
        return Err("CVPixelBuffer base address is null".into());
    }
    
    // Copy pixel data from locked buffer
    let raw_data = std::slice::from_raw_parts(
        base_address as *const u8,
        buffer_size
    );
    
    // Convert based on detected pixel format
    match pixel_format {
        kCVPixelFormatType_32BGRA => convert_bgra_to_rgba(raw_data, width, height, bytes_per_row),
        kCVPixelFormatType_32ARGB => convert_argb_to_rgba(raw_data, width, height, bytes_per_row),
        kCVPixelFormatType_24RGB => convert_rgb_to_rgba(raw_data, width, height, bytes_per_row),
        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange => {
            convert_yuv420_to_rgba(frame, width, height)
        }
        _ => Err(format!("Unsupported pixel format: {:?}", pixel_format).into()),
    }
}
```

### Pixel Format Conversion Functions

```rust
fn convert_bgra_to_rgba(
    data: &[u8], 
    width: usize, 
    height: usize, 
    bytes_per_row: usize
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut rgba_buffer = Vec::with_capacity(width * height * 4);
    
    // Handle row padding - similar to screenshots-rs pattern
    for row in data.chunks_exact(bytes_per_row) {
        let row_data = &row[..width * 4]; // Remove padding
        rgba_buffer.extend_from_slice(row_data);
    }
    
    // BGRA -> RGBA conversion (swap R and B channels)
    for bgra in rgba_buffer.chunks_exact_mut(4) {
        bgra.swap(0, 2); // B <-> R
    }
    
    Ok(rgba_buffer)
}

fn convert_argb_to_rgba(
    data: &[u8], 
    width: usize, 
    height: usize, 
    bytes_per_row: usize
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut rgba_buffer = Vec::with_capacity(width * height * 4);
    
    for row in data.chunks_exact(bytes_per_row) {
        let row_data = &row[..width * 4];
        for argb in row_data.chunks_exact(4) {
            // ARGB -> RGBA reordering: A,R,G,B -> R,G,B,A
            rgba_buffer.push(argb[1]); // R
            rgba_buffer.push(argb[2]); // G
            rgba_buffer.push(argb[3]); // B
            rgba_buffer.push(argb[0]); // A
        }
    }
    
    Ok(rgba_buffer)
}

fn convert_rgb_to_rgba(
    data: &[u8], 
    width: usize, 
    height: usize, 
    bytes_per_row: usize
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut rgba_buffer = Vec::with_capacity(width * height * 4);
    
    for row in data.chunks_exact(bytes_per_row) {
        let row_data = &row[..width * 3]; // RGB has 3 bytes per pixel
        for rgb in row_data.chunks_exact(3) {
            rgba_buffer.push(rgb[0]); // R
            rgba_buffer.push(rgb[1]); // G
            rgba_buffer.push(rgb[2]); // B
            rgba_buffer.push(255);    // A (full opacity)
        }
    }
    
    Ok(rgba_buffer)
}
```

## Revised Implementation Steps (Infrastructure-Aware)

### 1. **Implement CVPixelBuffer Locking Pattern** (1.5 hours)
**File**: [`packages/livekit/src/playback.rs:746-754`](../../packages/livekit/src/playback.rs#L746)
```rust
// REPLACE get_buffer_data() with proper CVPixelBuffer access
unsafe fn get_buffer_data(
    frame: &RemoteVideoFrame,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Use CVPixelBufferLockBaseAddress pattern for safe memory access
    // Extract real width/height from CVPixelBuffer
    // Get base address and bytes_per_row for proper data access
}
```

### 2. **Add Pixel Format Detection** (45 minutes)
```rust
// Add to get_buffer_data() 
let pixel_format = frame.pixel_format_type();
match pixel_format {
    kCVPixelFormatType_32BGRA => convert_bgra_to_rgba(...),
    kCVPixelFormatType_32ARGB => convert_argb_to_rgba(...),
    // ... other formats
}
```

### 3. **Implement Format Conversion Functions** (2.5 hours)
- **BGRA to RGBA**: Swap R and B channels + handle row padding  
- **ARGB to RGBA**: Reorder channels A,R,G,B -> R,G,B,A
- **RGB to RGBA**: Add alpha channel (255 for full opacity)
- **YUV420 to RGBA**: Planar YUV to interleaved RGB conversion

### 4. **Add Proper Error Handling** (1 hour)
```rust
#[derive(Debug, thiserror::Error)]
pub enum VideoFrameError {
    #[error("Unsupported pixel format: {format:?}")]
    UnsupportedFormat { format: u32 },
    
    #[error("CVPixelBuffer lock failed: {code}")]
    BufferLockFailed { code: i32 },
    
    #[error("Invalid frame dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    
    #[error("Buffer underrun: expected {expected} bytes, got {actual}")]
    BufferUnderrun { expected: usize, actual: usize },
}
```

### 5. **Add Dependencies for Core Video Constants** (15 minutes)
**File**: [`packages/livekit/Cargo.toml`](../../packages/livekit/Cargo.toml)
```toml
[dependencies]
# Core Video pixel format constants
core-foundation = "0.9"
```

### 6. **Testing and Validation** (1.5 hours)
- Unit tests for each conversion function with known input/output
- Integration tests with real RemoteVideoFrame data
- Error condition testing (unsupported formats, buffer failures)
- Memory safety testing (proper lock/unlock cycles)

## Available Research Resources

### Core Graphics Pattern Reference
**Primary Reference**: [`./tmp/screenshots-rs/src/macos/capture.rs`](../../tmp/screenshots-rs/src/macos/capture.rs)  
- Complete CGDataProvider data extraction pattern
- BGRA->RGBA conversion with row padding handling  
- Error handling for capture failures
- Memory-efficient buffer allocation patterns

### Pixel Format Constants  
**Reference**: [`./tmp/core-video/core-video/src/pixel_buffer.rs:25-78`](../../tmp/core-video/core-video/src/pixel_buffer.rs#L25)
- Comprehensive CVPixelFormat constants (kCVPixelFormatType_32BGRA, etc.)
- Planar vs packed format identification
- Color space and bit depth specifications

### Existing Video Infrastructure Integration
**Reference**: [`packages/video/src/macos.rs:280-315`](../../packages/video/src/macos.rs#L280)  
- MacOSVideoFrame CVPixelBuffer integration patterns
- ThreadSafeCVImageBuffer wrapper usage
- VideoFrameImpl trait implementation patterns

## Success Criteria

- [ ] **No Black Frames**: `to_rgba_bytes()` returns actual pixel data instead of black placeholders  
- [ ] **Real Video Content**: RemoteVideoFrame CVPixelBuffer data is properly extracted and converted
- [ ] **Multiple Format Support**: BGRA, ARGB, RGB formats convert correctly to RGBA
- [ ] **Memory Safety**: Proper CVPixelBuffer locking/unlocking with no leaks
- [ ] **Error Handling**: Meaningful error messages for unsupported formats and buffer failures
- [ ] **Performance**: Real-time video frame extraction with <10ms latency per frame
- [ ] **Integration**: Works seamlessly with existing VideoFrameExtensions trait

## Platform Support Matrix

| Platform | Status | Implementation | Dependencies |
|----------|--------|----------------|-----------------|
| macOS | ✅ **Complete** | CVPixelBuffer extraction + format conversion | core-foundation |
| Linux | 🔧 **Stub** | Generic buffer handling (future) | None |
| Windows | 🔧 **Stub** | Generic buffer handling (future) | None |

## Risk Assessment

**Risk Level**: MEDIUM - Infrastructure exists, only need data extraction fix  
**Revised Effort Estimate**: 6-8 hours (reduced from 8-10 due to existing infrastructure)  
**Complexity**: Medium (CVPixelBuffer API + pixel format handling)

**Risks**:
- CVPixelBuffer API complexity and memory management
- Pixel format detection accuracy 
- Performance impact from format conversions
- Thread safety with buffer locking

**Mitigations**:
- Use defer patterns for guaranteed buffer unlocking
- Test with multiple pixel formats and real video data
- Profile conversion performance during development
- Follow established ThreadSafeCVImageBuffer patterns

## Completion Definition

Task is complete when:
1. ✅ `cargo check --package livekit` passes without warnings
2. ✅ No black frame placeholders remain - `get_buffer_data()` extracts real CVPixelBuffer data  
3. ✅ All supported pixel formats (BGRA, ARGB, RGB) convert correctly to RGBA
4. ✅ CVPixelBuffer locking/unlocking works safely without memory leaks
5. ✅ Error handling provides clear messages for unsupported formats and buffer failures
6. ✅ Performance meets real-time video processing requirements (<10ms per frame)
7. ✅ Integration with existing VideoFrameExtensions trait maintains compatibility
8. ✅ RemoteVideoFrame displays real video content instead of black screens

## Dependencies Resolution

**Before Starting**: CVPixelBuffer type definitions are available (already exists)  
**Parallel Work**: Can develop alongside other video tasks, provides foundation for video quality  
**Success Enabler**: Critical for all remote video display functionality in LiveKit integration

## Implementation Priority

This task fixes a critical silent failure that makes video appear broken to users. With the infrastructure already in place, this becomes a **focused 6-8 hour implementation** to replace the black placeholder with proper CVPixelBuffer data extraction.

**Key Insight**: The hard work (trait definitions, thread safety, integration) is already done. We just need to replace `vec![0u8; buffer_size]` with proper `CVPixelBufferLockBaseAddress` + format conversion calls.