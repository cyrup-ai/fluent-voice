# Task: Implement VideoSource Integration for Camera and Screen Capture

**Priority**: ⚠️ HIGH (Core Functionality)  
**File**: [`packages/video/src/main.rs:141-145`](../../packages/video/src/main.rs#L141)  
**Milestone**: 1_video_processing_infrastructure  
**Status**: ✅ **VERIFIED CURRENT** - Extensive codebase analysis and reference research confirms implementation plan

## Problem Description

VideoSource creation needs proper platform-specific implementations to replace test pattern generation:

```rust
// CURRENT (TEST PATTERNS ONLY - VERIFIED):
let source = if camera {
    VideoSource::from_camera(options).context("Failed to create camera source")?
} else {
    VideoSource::from_screen(options).context("Failed to create screen source")?
};
```

**Current Implementation Analysis - [`packages/video/src/macos.rs:335-375`](../../packages/video/src/macos.rs#L335)**:
```rust
// CONFIRMED: Only generates test patterns, no real device access
fn start(&mut self) -> Result<()> {
    // Start video capture session and frame generation
    // Initiates frame capture loop with test pattern generation
    let frame = macos_source.generate_test_pattern(frame_number); // ← TEST PATTERN
}
```

**Impact**: 
- VideoSource::from_camera() and from_screen() only generate test patterns (verified)
- No real AVFoundation or ScreenCaptureKit integration despite existing framework
- Missing device enumeration and selection capabilities
- CVPixelBuffer integration exists but unused [`packages/video/src/macos.rs:76-89`](../../packages/video/src/macos.rs#L76)
- Perfect infrastructure exists, just needs real device connections

## Major Discovery: Implementation Infrastructure is Complete! ⚡

**CRITICAL FINDING**: The VideoSource architecture and CVPixelBuffer integration infrastructure is already complete and production-ready! Only device integration is missing.

### Existing Working Infrastructure

**1. Complete MacOS CVPixelBuffer Framework - Ready** [`packages/video/src/macos.rs:47-135`](../../packages/video/src/macos.rs#L47):
```rust
// COMPLETE CVPixelBuffer integration framework (unused):
impl MacOSVideoFrame {
    /// Create from CVImageBuffer with real Core Video dimensions
    pub fn from_cv_buffer(buffer: CVImageBuffer, timestamp_us: i64) -> Self {
        let display_size = buffer.get_display_size();
        let width = display_size.width as u32;
        let height = display_size.height as u32;
        
        Self {
            native: None,
            buffer: Some(ThreadSafeCVImageBuffer::new(buffer)), // ← READY FOR REAL DATA
            width, height, timestamp_us,
        }
    }
    
    /// Thread-safe CVPixelBuffer access
    pub fn cv_buffer(&self) -> Option<&CVImageBuffer> {
        self.buffer.as_ref().map(|b| b.get())
    }
}
```

**2. Platform Abstraction Layer - Complete** [`packages/video/src/video_source.rs:15-73`](../../packages/video/src/video_source.rs#L15):
```rust
// COMPLETE platform-specific dispatch (works perfectly):
impl VideoSource {
    pub fn from_camera(options: VideoSourceOptions) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let inner = MacOSVideoSource::from_camera(options)?; // ← Only needs AVFoundation
            Ok(Self { inner: Arc::new(Mutex::new(inner)) })
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            let inner = GenericVideoSource::from_camera(options)?; // ← Cross-platform fallback
        }
    }
}
```

**3. Dependencies Already Configured** [`packages/video/Cargo.toml:133-138`](../../packages/video/Cargo.toml#L133):
```toml
# Core Video framework already included:
[target.'cfg(target_os = "macos")'.dependencies]
core-foundation = "0.10.1"
core-video = "0.4.3"         # ← ALREADY PRESENT
coreaudio-rs = "0.13.0"
objc2 = "0.6"
```

**4. Cross-Platform Fallback Ready** [`packages/video/src/generic.rs:58-117`](../../packages/video/src/generic.rs#L58):
```rust
// Generic implementation ready for Linux/Windows integration:
impl GenericVideoSource {
    pub fn from_camera(options: VideoSourceOptions) -> Result<Self> {
        // Platform stub ready for V4L2/DirectShow integration
    }
    
    pub fn from_screen(options: VideoSourceOptions) -> Result<Self> {
        // Platform stub ready for X11/DXGI integration  
    }
}
```

### What's Actually Missing (Small Implementation Gap)

**ONLY Missing**: Replace test pattern generation with real device capture in platform implementations

## Complete Reference Implementation Patterns

### AVFoundation Camera Capture Pattern

**Complete Working Reference**: [`./tmp/core-video/av-foundation/examples/video_capture.rs:75-153`](../../tmp/core-video/av-foundation/examples/video_capture.rs#L75)

```rust
use av_foundation::{
    capture_device::{
        AVCaptureDevice, AVCaptureDeviceDiscoverySession, 
        AVCaptureDeviceTypeBuiltInWideAngleCamera, AVCaptureDeviceTypeExternal
    },
    capture_input::AVCaptureDeviceInput,
    capture_session::AVCaptureSession,
    capture_video_data_output::{AVCaptureVideoDataOutput, AVCaptureVideoDataOutputSampleBufferDelegate},
    media_format::AVMediaTypeVideo,
};

// DEVICE DISCOVERY PATTERN:
fn enumerate_camera_devices() -> Result<Vec<AVCaptureDevice>> {
    let mut device_types = NSMutableArray::new();
    device_types.addObject(AVCaptureDeviceTypeBuiltInWideAngleCamera);
    device_types.addObject(AVCaptureDeviceTypeExternal);
    
    let devices = AVCaptureDeviceDiscoverySession::discovery_session_with_device_types(
        &device_types,
        AVMediaTypeVideo,
        AVCaptureDevicePositionUnspecified,
    ).devices();
    
    Ok(devices)
}

// CAPTURE SESSION PATTERN:
struct CameraCaptureDelegate {
    frame_sender: Arc<Mutex<Option<VideoFrameSender>>>,
}

impl AVCaptureVideoDataOutputSampleBufferDelegate for CameraCaptureDelegate {
    fn capture_output_did_output_sample_buffer(
        &self,
        _capture_output: &AVCaptureOutput,
        sample_buffer: CMSampleBufferRef,
        _connection: &AVCaptureConnection,
    ) {
        let sample_buffer = CMSampleBuffer::wrap_under_get_rule(sample_buffer);
        if let Some(image_buffer) = sample_buffer.get_image_buffer() {
            if let Some(pixel_buffer) = image_buffer.downcast::<CVPixelBuffer>() {
                // Create MacOSVideoFrame directly from CVPixelBuffer:
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as i64;
                
                let video_frame = MacOSVideoFrame::from_cv_buffer(pixel_buffer, timestamp);
                let frame = VideoFrame::new(video_frame);
                
                // Send frame to VideoSource current_frame
                if let Some(sender) = &*self.frame_sender.lock().unwrap() {
                    sender.send(frame).ok();
                }
            }
        }
    }
}

// COMPLETE CAMERA SOURCE SETUP:
pub fn create_camera_source(device_id: Option<String>, options: VideoSourceOptions) -> Result<MacOSVideoSource> {
    // 1. Enumerate devices
    let devices = enumerate_camera_devices()?;
    
    // 2. Select device
    let device = if let Some(id) = &device_id {
        devices.iter().find(|d| d.unique_id() == *id)
    } else {
        devices.first()
    }.ok_or(VideoSourceError::CameraNotFound)?;
    
    // 3. Create capture session
    let session = AVCaptureSession::new();
    let input = AVCaptureDeviceInput::from_device(device)?;
    let output = AVCaptureVideoDataOutput::new();
    
    // 4. Configure output delegate
    let delegate = CameraCaptureDelegate { frame_sender };
    let queue = Queue::new("com.fluent_video.camera", QueueAttribute::Serial);
    output.set_sample_buffer_delegate(&delegate, &queue);
    
    // 5. Assemble pipeline
    session.begin_configuration();
    session.add_input(&input);
    session.add_output(&output);
    session.commit_configuration();
    
    // 6. Return configured source with real capture
    Ok(MacOSVideoSource {
        info: VideoSourceInfo { /* real device info */ },
        capture_session: Some(session), // ← REAL CAPTURE SESSION
        // Remove frame_timer test pattern generation
    })
}
```

### ScreenCaptureKit Screen Capture Pattern

**Complete Working Reference**: [`./tmp/core-video/screen-capture-kit/examples/screen_capture.rs:75-119`](../../tmp/core-video/screen-capture-kit/examples/screen_capture.rs#L75)

```rust
use screen_capture_kit::{
    shareable_content::SCShareableContent,
    stream::{SCContentFilter, SCStream, SCStreamConfiguration, 
             SCStreamDelegate, SCStreamOutput, SCStreamOutputType},
};

// DISPLAY DISCOVERY PATTERN:
fn enumerate_displays() -> Result<Vec<SCDisplay>> {
    let (tx, rx) = channel();
    SCShareableContent::get_shareable_content_with_completion_closure(move |content, error| {
        let result = content.ok_or_else(|| error.unwrap());
        tx.send(result).unwrap();
    });
    
    let shareable_content = rx.recv()??;
    Ok(shareable_content.displays())
}

// SCREEN CAPTURE DELEGATE PATTERN:
struct ScreenCaptureDelegate {
    frame_sender: Arc<Mutex<Option<VideoFrameSender>>>,
}

impl SCStreamOutput for ScreenCaptureDelegate {
    fn stream_did_output_sample_buffer(
        &self, 
        _stream: &SCStream, 
        sample_buffer: CMSampleBufferRef, 
        of_type: SCStreamOutputType
    ) {
        if of_type != SCStreamOutputType::Screen { return; }
        
        let sample_buffer = CMSampleBuffer::wrap_under_get_rule(sample_buffer);
        if let Some(image_buffer) = sample_buffer.get_image_buffer() {
            if let Some(pixel_buffer) = image_buffer.downcast::<CVPixelBuffer>() {
                // Direct integration with existing MacOSVideoFrame:
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as i64;
                
                let video_frame = MacOSVideoFrame::from_cv_buffer(pixel_buffer, timestamp);
                let frame = VideoFrame::new(video_frame);
                
                // Send to VideoSource pipeline
                if let Some(sender) = &*self.frame_sender.lock().unwrap() {
                    sender.send(frame).ok();
                }
            }
        }
    }
}

// COMPLETE SCREEN SOURCE SETUP:
pub fn create_screen_source(display_id: Option<u32>, options: VideoSourceOptions) -> Result<MacOSVideoSource> {
    // 1. Enumerate displays
    let displays = enumerate_displays()?;
    
    // 2. Select display
    let display = if let Some(id) = display_id {
        displays.iter().find(|d| d.display_id() == id)
    } else {
        displays.first()
    }.ok_or(VideoSourceError::DisplayNotFound)?;
    
    // 3. Create stream configuration
    let filter = SCContentFilter::init_with_display_exclude_windows(
        SCContentFilter::alloc(), display, &NSArray::new()
    );
    let configuration = SCStreamConfiguration::new();
    configuration.set_width(options.width.unwrap_or(display.width()));
    configuration.set_height(options.height.unwrap_or(display.height()));
    
    // 4. Create capture stream
    let delegate = ScreenCaptureDelegate { frame_sender };
    let stream = SCStream::init_with_filter(&filter, &configuration, &delegate);
    
    // 5. Start capture
    let queue = Queue::new("com.fluent_video.screen", QueueAttribute::Serial);
    stream.add_stream_output(&delegate, SCStreamOutputType::Screen, &queue)?;
    
    Ok(MacOSVideoSource {
        info: VideoSourceInfo { /* real display info */ },
        screen_capture_stream: Some(stream), // ← REAL SCREEN CAPTURE
        // Remove frame_timer test pattern generation  
    })
}
```

### Cross-Platform Device Enumeration Pattern

**Complete Working Reference**: [`./tmp/screenshots-rs/examples/monitor.rs:4-25`](../../tmp/screenshots-rs/examples/monitor.rs#L4)

```rust
use xcap::Monitor; // Cross-platform monitor enumeration

// CROSS-PLATFORM DISPLAY ENUMERATION:
pub fn enumerate_video_devices() -> Result<Vec<VideoDevice>> {
    let mut devices = Vec::new();
    
    #[cfg(target_os = "macos")]
    {
        // Camera enumeration using AVFoundation
        let cameras = enumerate_camera_devices()?;
        for camera in cameras {
            devices.push(VideoDevice {
                id: camera.unique_id(),
                name: camera.localized_name(),
                device_type: VideoDeviceType::Camera,
                platform_data: PlatformDeviceData::AVCaptureDevice(camera),
            });
        }
        
        // Display enumeration using ScreenCaptureKit  
        let displays = enumerate_displays()?;
        for display in displays {
            devices.push(VideoDevice {
                id: display.display_id().to_string(),
                name: display.localized_name(),
                device_type: VideoDeviceType::Display,
                platform_data: PlatformDeviceData::SCDisplay(display),
            });
        }
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        // Generic cross-platform display enumeration
        let monitors = Monitor::all()?;
        for monitor in monitors {
            devices.push(VideoDevice {
                id: monitor.id()?.to_string(),
                name: monitor.name()?,
                device_type: VideoDeviceType::Display,
                resolution: (monitor.width()?, monitor.height()?),
                platform_data: PlatformDeviceData::Monitor(monitor),
            });
        }
    }
    
    Ok(devices)
}
```

## Enhanced Implementation Plan (Infrastructure-Aware)

### 1. **Replace Test Pattern with Real AVFoundation Camera** (3 hours)
**File**: [`packages/video/src/macos.rs:200-223`](../../packages/video/src/macos.rs#L200)

```rust
impl MacOSVideoSource {
    pub fn from_camera(options: VideoSourceOptions) -> Result<Self> {
        // REPLACE: Current test pattern initialization
        
        // ADD: Real AVFoundation camera setup
        let capture_session = Self::create_camera_capture_session(&options)?;
        
        Ok(Self {
            info: VideoSourceInfo {
                name: "AVFoundation Camera".to_string(),
                width: options.width.unwrap_or(1280),
                height: options.height.unwrap_or(720),
                fps: options.fps.unwrap_or(30) as f64,
                is_active: false,
            },
            current_frame: Arc::new(RwLock::new(None)),
            is_active: false,
            
            // REMOVE: frame_timer test pattern generation
            // ADD: Real capture session
            av_capture_session: Some(capture_session), 
            capture_session_id: Some(nanoid::nanoid!()),
            capture_device_id: options.device_id,
        })
    }
    
    fn create_camera_capture_session(options: &VideoSourceOptions) -> Result<AVCaptureSession> {
        // Implementation using patterns from ./tmp/core-video/av-foundation/examples/video_capture.rs
        // Full pattern shown above in reference implementation
    }
}
```

### 2. **Replace Test Pattern with Real ScreenCaptureKit Screen** (3 hours)
**File**: [`packages/video/src/macos.rs:225-248`](../../packages/video/src/macos.rs#L225)

```rust
impl MacOSVideoSource {
    pub fn from_screen(options: VideoSourceOptions) -> Result<Self> {
        // REPLACE: Current test pattern initialization
        
        // ADD: Real ScreenCaptureKit screen setup
        let screen_stream = Self::create_screen_capture_stream(&options)?;
        
        Ok(Self {
            info: VideoSourceInfo {
                name: "ScreenCaptureKit Display".to_string(),
                width: options.width.unwrap_or(1920),  
                height: options.height.unwrap_or(1080),
                fps: options.fps.unwrap_or(30) as f64,
                is_active: false,
            },
            current_frame: Arc::new(RwLock::new(None)),
            is_active: false,
            
            // REMOVE: frame_timer test pattern generation
            // ADD: Real screen capture stream
            screen_capture_stream: Some(screen_stream),
            capture_session_id: Some(nanoid::nanoid!()),
            capture_device_id: options.device_id.or(Some("main_display".to_string())),
        })
    }
    
    fn create_screen_capture_stream(options: &VideoSourceOptions) -> Result<SCStream> {
        // Implementation using patterns from ./tmp/core-video/screen-capture-kit/examples/screen_capture.rs  
        // Full pattern shown above in reference implementation
    }
}
```

### 3. **Enhance VideoSourceOptions with Device Selection** (1 hour)
**File**: [`packages/video/src/video_source.rs:23-34`](../../packages/video/src/video_source.rs#L23)

```rust
#[derive(Debug, Clone, Default)]
pub struct VideoSourceOptions {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<u32>,
    
    // ADD: Device selection and format options
    pub device_id: Option<String>,           // Camera ID or display ID
    pub pixel_format: Option<PixelFormat>,   // Preferred pixel format (YUV, RGB, etc)
    pub buffer_count: Option<u32>,           // Buffer management hint
    pub low_latency: bool,                   // Performance vs quality tradeoff
}

#[derive(Debug, Clone, Copy)]
pub enum PixelFormat {
    BGRA32,   // Core Video default  
    YUV420,   // Efficient for encoding
    RGB24,    // Standard RGB
    NV12,     // Hardware-friendly YUV
}
```

### 4. **Add Device Enumeration API** (2 hours)
**New File**: `packages/video/src/device_enumeration.rs`

```rust
#[derive(Debug, Clone)]
pub struct VideoDevice {
    pub id: String,
    pub name: String,
    pub device_type: VideoDeviceType,
    pub resolution: (u32, u32),
    pub supported_formats: Vec<PixelFormat>,
    pub is_default: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum VideoDeviceType {
    Camera,
    Display,
    VirtualCamera,
}

/// Enumerate all available video devices (cameras + displays)
pub fn enumerate_video_devices() -> Result<Vec<VideoDevice>> {
    // Platform-specific implementation using patterns from research
    // Full implementation shown above in reference patterns
}

/// Get default camera device
pub fn get_default_camera() -> Result<Option<VideoDevice>> {
    Ok(enumerate_video_devices()?.into_iter()
        .find(|d| d.device_type == VideoDeviceType::Camera && d.is_default))
}

/// Get primary display device  
pub fn get_primary_display() -> Result<Option<VideoDevice>> {
    Ok(enumerate_video_devices()?.into_iter()
        .find(|d| d.device_type == VideoDeviceType::Display && d.is_default))
}
```

### 5. **Implement Comprehensive Error Handling** (1.5 hours)
**File**: [`packages/video/src/video_source.rs:171+`](../../packages/video/src/video_source.rs#L171)

```rust
#[derive(Debug, thiserror::Error)]
pub enum VideoSourceError {
    #[error("Camera device not found: {device_id}")]
    CameraNotFound { device_id: String },
    
    #[error("Display not found: {display_id}")]
    DisplayNotFound { display_id: String },
    
    #[error("Permission denied for camera access")]
    CameraPermissionDenied,
    
    #[error("Permission denied for screen recording")]
    ScreenRecordingPermissionDenied,
    
    #[error("Unsupported resolution: {width}x{height} for device {device_id}")]
    UnsupportedResolution { width: u32, height: u32, device_id: String },
    
    #[error("Device busy: {device_id} is already in use")]
    DeviceBusy { device_id: String },
    
    #[error("AVFoundation error: {message}")]
    AVFoundationError { message: String },
    
    #[error("ScreenCaptureKit error: {message}")]  
    ScreenCaptureKitError { message: String },
    
    #[error("Core Video error: {message}")]
    CoreVideoError { message: String },
}

// Permission handling for macOS
pub async fn request_camera_permission() -> Result<bool> {
    // Implementation using AVCaptureDevice::authorization_status_for_media_type
}

pub async fn request_screen_recording_permission() -> Result<bool> {
    // Implementation using CGPreflightScreenCaptureAccess
}
```

### 6. **Linux/Windows Cross-Platform Stubs** (2 hours)  
**New Files**: `packages/video/src/platform/linux.rs`, `packages/video/src/platform/windows.rs`

```rust
// Linux V4L2 camera stub (packages/video/src/platform/linux.rs):
pub fn enumerate_camera_devices() -> Result<Vec<VideoDevice>> {
    // TODO: Implement V4L2 camera enumeration
    // Reference: https://www.kernel.org/doc/html/latest/media/uapi/v4l/v4l2.html
    // Use v4l crate: https://crates.io/crates/v4l
    Ok(Vec::new())
}

// Windows DirectShow camera stub (packages/video/src/platform/windows.rs):
pub fn enumerate_camera_devices() -> Result<Vec<VideoDevice>> {
    // TODO: Implement DirectShow camera enumeration  
    // Reference: https://docs.microsoft.com/en-us/windows/win32/directshow/
    // Use windows crate: https://crates.io/crates/windows
    Ok(Vec::new())
}
```

### 7. **Update Dependencies** (30 minutes)
**File**: [`packages/video/Cargo.toml`](../../packages/video/Cargo.toml)

```toml
# ADD: Required dependencies for real device integration
[target.'cfg(target_os = "macos")'.dependencies]
# Already present: core-foundation = "0.10.1", core-video = "0.4.3", objc2 = "0.6"

# ADD: AVFoundation and ScreenCaptureKit bindings
av-foundation = "0.4"          # Camera capture
screen-capture-kit = "0.3"     # Screen capture  
dispatch2 = "0.5"              # GCD queue management
nanoid = "0.4"                 # Unique session IDs

[target.'cfg(target_os = "linux")'.dependencies]  
v4l = { version = "0.15", optional = true }       # Video4Linux camera access
xcap = "0.0.12"                                   # Cross-platform screen capture

[target.'cfg(target_os = "windows')'.dependencies]
windows = { version = "0.58", optional = true, features = [
    "Media_Devices",           # Camera enumeration
    "Graphics_Capture",        # Screen capture
] }

[features]
# ADD: Platform-specific features
linux-camera = ["v4l"]
windows-camera = ["windows"]
```

### 8. **Integration Testing and Validation** (2 hours)
**New File**: `packages/video/tests/device_integration.rs`

```rust
#[tokio::test]
async fn test_camera_enumeration() {
    let devices = enumerate_video_devices().unwrap();
    let cameras: Vec<_> = devices.iter()
        .filter(|d| d.device_type == VideoDeviceType::Camera)
        .collect();
    
    if !cameras.is_empty() {
        assert!(!cameras[0].id.is_empty());
        assert!(!cameras[0].name.is_empty());
        println!("Found camera: {} ({})", cameras[0].name, cameras[0].id);
    } else {
        println!("No cameras found (acceptable in CI/headless environments)");
    }
}

#[tokio::test]  
async fn test_real_camera_source_creation() {
    let devices = enumerate_video_devices().unwrap();
    let camera = devices.iter()
        .find(|d| d.device_type == VideoDeviceType::Camera);
        
    if let Some(camera_device) = camera {
        let options = VideoSourceOptions {
            width: Some(640),
            height: Some(480), 
            fps: Some(30),
            device_id: Some(camera_device.id.clone()),
            ..Default::default()
        };
        
        let result = VideoSource::from_camera(options);
        assert!(result.is_ok(), "Camera source creation should succeed with real device");
        
        let source = result.unwrap();
        source.start().unwrap();
        
        // Verify real frames (not test patterns)
        tokio::time::sleep(Duration::from_millis(500)).await;
        let frame = source.get_current_frame();
        assert!(frame.is_some(), "Should receive real camera frames");
        
        let frame = frame.unwrap();
        assert!(frame.width() > 0 && frame.height() > 0);
        assert!(frame.timestamp_us() > 0, "Real frames have timestamps");
        
        source.stop().unwrap();
    } else {
        println!("Skipping camera test - no cameras available");
    }
}

#[tokio::test]
async fn test_real_screen_source_creation() {
    let devices = enumerate_video_devices().unwrap();
    let display = devices.iter()
        .find(|d| d.device_type == VideoDeviceType::Display);
        
    if let Some(display_device) = display {
        let options = VideoSourceOptions {
            width: Some(1280),
            height: Some(720),
            fps: Some(30), 
            device_id: Some(display_device.id.clone()),
            ..Default::default()
        };
        
        let result = VideoSource::from_screen(options);
        assert!(result.is_ok(), "Screen source creation should succeed with real display");
        
        let source = result.unwrap();
        source.start().unwrap();
        
        // Verify real screen capture (not test patterns)
        tokio::time::sleep(Duration::from_millis(500)).await;
        let frame = source.get_current_frame();
        assert!(frame.is_some(), "Should receive real screen frames");
        
        source.stop().unwrap();
    } else {
        println!("Skipping screen test - no displays available");
    }
}

#[tokio::test]
async fn test_camera_to_video_track_pipeline() {
    // Test end-to-end pipeline with real devices
    if let Ok(devices) = enumerate_video_devices() {
        if let Some(camera) = devices.iter().find(|d| d.device_type == VideoDeviceType::Camera) {
            let options = VideoSourceOptions {
                device_id: Some(camera.id.clone()),
                ..Default::default()
            };
            
            let source = VideoSource::from_camera(options).unwrap();
            let track = VideoTrack::new(source);
            
            track.play().unwrap();
            tokio::time::sleep(Duration::from_millis(200)).await;
            
            let frame = track.get_current_frame();
            assert!(frame.is_some(), "VideoTrack should receive real frames from VideoSource");
            assert!(!frame.unwrap().is_empty());
        }
    }
}
```

## Available Research Resources

### Core Video Integration Examples
**Primary Reference**: [`./tmp/core-video/av-foundation/examples/video_capture.rs`](../../tmp/core-video/av-foundation/examples/video_capture.rs)
- Complete AVFoundation device discovery with `AVCaptureDeviceDiscoverySession`
- Capture session setup with `AVCaptureSession`, `AVCaptureDeviceInput`, `AVCaptureVideoDataOutput`
- Sample buffer delegate pattern with `AVCaptureVideoDataOutputSampleBufferDelegate`  
- CVPixelBuffer extraction from `CMSampleBuffer` frames
- Device enumeration by type (built-in cameras, external cameras)
- Queue management with `dispatch2::Queue` for thread safety

**ScreenCaptureKit Reference**: [`./tmp/core-video/screen-capture-kit/examples/screen_capture.rs`](../../tmp/core-video/screen-capture-kit/examples/screen_capture.rs)
- Display discovery with `SCShareableContent::get_shareable_content`
- Stream configuration with `SCStreamConfiguration` (resolution, format)
- Content filtering with `SCContentFilter::init_with_display_exclude_windows`
- Stream delegate pattern with `SCStreamOutput` and `SCStreamDelegate`
- Direct CVPixelBuffer delivery to existing `MacOSVideoFrame::from_cv_buffer`

### Cross-Platform Device Enumeration
**Primary Reference**: [`./tmp/screenshots-rs/examples/monitor.rs`](../../tmp/screenshots-rs/examples/monitor.rs)
- Cross-platform display enumeration with `xcap::Monitor::all()`
- Display properties: `id()`, `name()`, `width()`, `height()`, `is_primary()`
- Display selection by coordinates with `Monitor::from_point(x, y)`
- Screen capture capabilities with position and scale factor detection

**Linux Screen Capture**: [`packages/video/Cargo.toml:131`](../../packages/video/Cargo.toml#L131)
- Already configured: `scap = "0.0.8"` for Linux screen capture
- Ready for V4L2 camera integration with optional `v4l` crate dependency

### Existing MacOS CVPixelBuffer Integration
**Current Framework**: [`packages/video/src/macos.rs:75-135`](../../packages/video/src/macos.rs#L75)
- Complete `MacOSVideoFrame::from_cv_buffer()` implementation
- Thread-safe `ThreadSafeCVImageBuffer` wrapper with Send + Sync
- RGBA conversion from CVPixelBuffer with `get_buffer_data()`
- Real Core Video dimension extraction with `buffer.get_display_size()`
- Integration with `VideoFrame::new()` for pipeline compatibility

## Success Criteria

- [x] **Infrastructure Analysis Complete**: MacOS CVPixelBuffer framework ready, platform abstraction working
- [ ] **Real Camera Capture**: AVFoundation camera → CVPixelBuffer → MacOSVideoFrame pipeline working  
- [ ] **Real Screen Capture**: ScreenCaptureKit display → CVPixelBuffer → MacOSVideoFrame pipeline working
- [ ] **Device Enumeration**: Camera and display discovery with device selection by ID
- [ ] **Enhanced Options**: VideoSourceOptions with device_id, pixel_format, performance hints
- [ ] **Permission Handling**: Camera and screen recording permission requests on macOS
- [ ] **Error Handling**: Meaningful errors for device not found, permission denied, device busy
- [ ] **Cross-Platform Stubs**: Linux V4L2 and Windows DirectShow stubs for future implementation
- [ ] **Integration Ready**: Works with existing VideoTrack → LiveKit publishing pipeline
- [ ] **Test Coverage**: Real device testing with fallbacks for headless environments

## Platform Support Matrix

| Platform | Camera | Screen | Implementation | Dependencies |
|----------|---------|---------|---------------|--------------|
| macOS | ✅ AVFoundation | ✅ ScreenCaptureKit | Production Ready | av-foundation, screen-capture-kit |
| Linux | 🔧 V4L2 Stub | ✅ xcap/scap | Partial Ready | v4l (optional), scap |
| Windows | 🔧 DirectShow Stub | 🔧 DXGI Stub | Stubs Only | windows crate (optional) |

## Risk Assessment

**Risk Level**: MODERATE (Infrastructure complete, only device integration needed)
**Revised Effort Estimate**: 10-12 hours (optimized due to complete CVPixelBuffer infrastructure)
**Complexity**: Medium-High (AVFoundation integration + permission handling)

**Critical Discovery**: The hard architectural work is done! CVPixelBuffer integration, thread safety, and platform abstraction are production-ready. Only need to replace test pattern generation with real device capture calls.

**Risks**:
- macOS permission dialogs may require user interaction during development
- AVFoundation callback lifecycle management requires careful resource cleanup
- Device availability testing across different macOS versions and hardware configs
- Performance optimization needed for real-time capture vs test pattern generation

**Mitigations**:
- Use existing working patterns from [`./tmp/core-video/av-foundation/examples/`](../../tmp/core-video/av-foundation/examples/)
- Implement graceful permission request flow with clear error messages
- Start with default devices, add device selection incrementally
- Test with multiple camera types (built-in, external, virtual)
- Reference [`MacOSVideoFrame::from_cv_buffer()`](../../packages/video/src/macos.rs#L76) for perfect CVPixelBuffer integration

## Completion Definition

Task is complete when:
1. ✅ MacOSVideoSource::from_camera() creates real AVFoundation camera sources (not test patterns)
2. ✅ MacOSVideoSource::from_screen() creates real ScreenCaptureKit screen sources (not test patterns)  
3. ✅ Device enumeration API works for both cameras and displays with detailed device info
4. ✅ VideoSourceOptions supports device selection, format preferences, and performance hints
5. ✅ Permission handling provides clear feedback and guidance for camera/screen recording access
6. ✅ Integration with existing VideoTrack → CVPixelBuffer → LiveKit pipeline produces real video
7. ✅ Cross-platform stubs provide clear implementation paths for Linux V4L2 and Windows DirectShow  
8. ✅ `cargo check --package fluent_video` passes without warnings
9. ✅ Integration tests demonstrate real camera and screen capture with fallbacks for headless environments

## Dependencies Resolution

**Before Starting**: VideoFrame CVPixelBuffer integration already complete and ready  
**Parallel Work**: Can develop alongside LiveKit track publishing (benefits from real frames)
**Critical Path**: Provides working VideoSource instances with real video data for all downstream tasks
**External Dependencies**: All required libraries available in [`./tmp/`](../../tmp/) for reference implementation

## Implementation Priority

This task transforms test pattern VideoSources into production-ready camera and screen capture sources. With complete CVPixelBuffer infrastructure and reference implementations available, this becomes a **focused 10-12 hour implementation** to enable real video capture throughout the fluent-voice ecosystem.

**Key Insight**: The infrastructure investment has already been made. We just need to call `session.start_running()` instead of `generate_test_pattern()`.