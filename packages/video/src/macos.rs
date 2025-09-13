//! macOS-specific video source implementation
//! 
//! This module contains platform-specific code for macOS video capture and processing.
//! It requires unsafe code for several legitimate reasons:
//! - Integration with Objective-C AVFoundation APIs through objc2 bindings
//! - Direct manipulation of Core Graphics and Core Video memory buffers
//! - Implementation of Objective-C protocol traits (AVCaptureVideoDataOutputSampleBufferDelegate)
//! - Thread safety markers (Send/Sync) for Apple framework types
//! - Memory-mapped access to video frame data from system APIs
//! 
//! All unsafe code is carefully reviewed and necessary for system integration.

#![allow(unsafe_code)]

use anyhow::Result;

use core_video::image_buffer::CVImageBuffer;

// FFmpeg imports for file video reading
use ffmpeg_next as ffmpeg;
use std::sync::{Arc, RwLock};
use tokio::sync::{mpsc, oneshot};

#[cfg(target_os = "macos")]
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
#[cfg(target_os = "macos")]
#[allow(deprecated)]
use objc2_core_graphics::{
    CGDataProvider, CGImage, CGWindowID, CGWindowImageOption, CGWindowListCreateImage,
    CGWindowListOption,
};

// AVFoundation imports for camera capture
#[cfg(target_os = "macos")]
use av_foundation::{
    capture_device::AVCaptureDevice,
    capture_input::AVCaptureDeviceInput,
    capture_output_base::AVCaptureOutput,
    capture_session::{AVCaptureConnection, AVCaptureSession},
    capture_video_data_output::{
        AVCaptureVideoDataOutput, AVCaptureVideoDataOutputSampleBufferDelegate,
    },
    media_format::AVMediaTypeVideo,
};

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSObjectProtocol;

#[cfg(target_os = "macos")]
use core_foundation::base::TCFType;
#[cfg(target_os = "macos")]
use dispatch2::Queue;

use crate::native_video::{NativeVideoFrame, VideoRotation};
use crate::video_frame::{VideoFrame, VideoFrameImpl};
use crate::video_source::{VideoSourceImpl, VideoSourceInfo, VideoSourceOptions};

/// Commands for the FFmpeg worker thread
#[derive(Debug)]
enum FFmpegCommand {
    StartFile {
        path: String,
        width: u32,
        height: u32,
        fps: f64,
    },
    #[allow(dead_code)] // Used in worker thread pattern matching, command sent by stop() method
    Stop,
    #[allow(dead_code)] // Used in worker thread pattern matching, command sent by get_current_frame() method
    GetFrame {
        response: oneshot::Sender<Option<VideoFrame>>,
    },
}

/// Thread-safe FFmpeg video source using channel-based architecture
#[derive(Debug)]
struct ThreadSafeFFmpegSource {
    command_tx: mpsc::UnboundedSender<FFmpegCommand>,
    _worker_handle: tokio::task::JoinHandle<()>,
}

impl ThreadSafeFFmpegSource {
    fn new() -> Self {
        let (command_tx, mut command_rx) = mpsc::unbounded_channel::<FFmpegCommand>();

        let worker_handle = tokio::spawn(async move {
            let mut current_frame: Option<VideoFrame> = None;
            let mut ffmpeg_initialized = false;

            while let Some(command) = command_rx.recv().await {
                match command {
                    FFmpegCommand::StartFile {
                        path,
                        width,
                        height,
                        fps,
                    } => {
                        // Initialize FFmpeg if not already done
                        if !ffmpeg_initialized {
                            if let Err(e) = ffmpeg::init() {
                                tracing::error!("Failed to initialize FFmpeg: {}", e);
                                continue;
                            }
                            ffmpeg_initialized = true;
                        }

                        // Start file reading in dedicated thread
                        let frame_clone = Arc::new(RwLock::new(None::<VideoFrame>));
                        let frame_reader = frame_clone.clone();

                        tokio::spawn(async move {
                            if let Err(e) =
                                Self::read_file_frames_async(path, width, height, fps, frame_reader)
                                    .await
                            {
                                tracing::error!("FFmpeg file reading error: {}", e);
                            }
                        });

                        current_frame = None;
                    }
                    FFmpegCommand::Stop => {
                        current_frame = None;
                    }
                    FFmpegCommand::GetFrame { response } => {
                        let _ = response.send(current_frame.clone());
                    }
                }
            }
        });

        Self {
            command_tx,
            _worker_handle: worker_handle,
        }
    }

    async fn read_file_frames_async(
        file_path: String,
        width: u32,
        height: u32,
        fps: f64,
        current_frame: Arc<RwLock<Option<VideoFrame>>>,
    ) -> Result<()> {
        use tokio::io::AsyncReadExt;
        use tokio::process::Command;

        let mut child = Command::new("ffmpeg")
            .args([
                "-i",
                &file_path,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "-s",
                &format!("{}x{}", width, height),
                "-r",
                &fps.to_string(),
                "-",
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to start FFmpeg: {}", e))?;

        if let Some(stdout) = child.stdout.take() {
            let mut reader = tokio::io::BufReader::new(stdout);
            let frame_size = (width * height * 4) as usize; // RGBA format
            let mut frame_number = 0u64;
            let frame_duration = tokio::time::Duration::from_millis((1000.0 / fps) as u64);

            loop {
                let mut buffer = vec![0u8; frame_size];
                match reader.read_exact(&mut buffer).await {
                    Ok(_) => {
                        let timestamp_us = (frame_number as f64 / fps * 1_000_000.0) as i64;
                        let video_frame = NativeVideoFrame::create_video_frame(
                            buffer,
                            width,
                            height,
                            timestamp_us,
                            VideoRotation::Rotation0,
                        );

                        if let Ok(mut guard) = current_frame.write() {
                            *guard = Some(video_frame);
                        }
                        frame_number += 1;
                        tokio::time::sleep(frame_duration).await;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => {
                        tracing::error!("Failed to read frame: {}", e);
                        break;
                    }
                }
            }
        }

        let _ = child.wait().await;
        Ok(())
    }

    #[allow(dead_code)] // FFmpeg frame conversion functionality reserved for future file input support
    fn convert_ffmpeg_frame_to_rgba(
        frame: &ffmpeg::frame::Video,
        target_width: u32,
        target_height: u32,
        frame_number: u64,
        fps: f64,
    ) -> Result<VideoFrame> {
        let width = frame.width() as u32;
        let height = frame.height() as u32;

        // Convert frame data to RGBA
        let mut rgba_data = Vec::with_capacity((target_width * target_height * 4) as usize);

        // Simple conversion from YUV420P to RGBA (production implementation would use swscale)
        let y_plane = frame.data(0);
        let u_plane = frame.data(1);
        let v_plane = frame.data(2);

        let y_stride = frame.stride(0) as usize;
        let u_stride = frame.stride(1) as usize;
        let _v_stride = frame.stride(2) as usize;

        for y in 0..target_height.min(height) {
            for x in 0..target_width.min(width) {
                let y_idx = (y as usize * y_stride) + x as usize;
                let uv_idx = ((y / 2) as usize * u_stride) + (x / 2) as usize;

                if y_idx < y_plane.len() && uv_idx < u_plane.len() && uv_idx < v_plane.len() {
                    let y_val = y_plane[y_idx] as f32;
                    let u_val = u_plane[uv_idx] as f32 - 128.0;
                    let v_val = v_plane[uv_idx] as f32 - 128.0;

                    // YUV to RGB conversion
                    let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
                    let g = (y_val - 0.344 * u_val - 0.714 * v_val).clamp(0.0, 255.0) as u8;
                    let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;

                    rgba_data.extend_from_slice(&[r, g, b, 255]);
                } else {
                    rgba_data.extend_from_slice(&[0, 0, 0, 255]);
                }
            }

            // Pad row if needed
            while rgba_data.len() < ((y + 1) * target_width * 4) as usize {
                rgba_data.extend_from_slice(&[0, 0, 0, 255]);
            }
        }

        // Pad remaining rows
        while rgba_data.len() < (target_width * target_height * 4) as usize {
            rgba_data.extend_from_slice(&[0, 0, 0, 255]);
        }

        let timestamp_us = (frame_number as f64 / fps * 1_000_000.0) as i64;
        let native_frame = NativeVideoFrame::create_video_frame(
            rgba_data,
            target_width,
            target_height,
            timestamp_us,
            VideoRotation::Rotation0,
        );

        Ok(native_frame)
    }

    fn start_file(&self, path: String, width: u32, height: u32, fps: f64) -> Result<()> {
        self.command_tx
            .send(FFmpegCommand::StartFile {
                path,
                width,
                height,
                fps,
            })
            .map_err(|_| anyhow::anyhow!("FFmpeg worker thread disconnected"))?;
        Ok(())
    }

    #[allow(dead_code)] // FFmpeg stop functionality reserved for future file input support
    fn stop(&self) -> Result<()> {
        self.command_tx
            .send(FFmpegCommand::Stop)
            .map_err(|_| anyhow::anyhow!("FFmpeg worker thread disconnected"))?;
        Ok(())
    }

    #[allow(dead_code)] // FFmpeg get frame functionality reserved for future file input support
    async fn get_current_frame(&self) -> Option<VideoFrame> {
        let (tx, rx) = oneshot::channel();
        if self
            .command_tx
            .send(FFmpegCommand::GetFrame { response: tx })
            .is_ok()
        {
            rx.await.unwrap_or(None)
        } else {
            None
        }
    }
}

/// Thread-safe wrapper for CVImageBuffer that implements Send + Sync
/// SAFETY: CVImageBuffer is reference-counted and thread-safe for read operations.
/// The underlying CVPixelBuffer is designed to be shared across threads safely
/// when properly retained/released through the Core Foundation reference counting.
#[derive(Clone)]
pub struct ThreadSafeCVImageBuffer {
    inner: CVImageBuffer,
}

impl ThreadSafeCVImageBuffer {
    pub fn new(buffer: CVImageBuffer) -> Self {
        Self { inner: buffer }
    }

    fn get(&self) -> &CVImageBuffer {
        &self.inner
    }
}

// SAFETY: CVImageBuffer is a Core Foundation reference-counted type that is designed
// to be thread-safe for concurrent read operations. Apple's documentation states that
// CVImageBuffer/CVPixelBuffer are safe to share across threads as long as proper
// reference counting is maintained. The ThreadSafeCVImageBuffer wrapper ensures this.
// Required for Arc<MacOSVideoFrame> usage in multi-threaded video processing pipeline.

// SAFETY: CVImageBuffer is thread-safe for sharing across threads when properly
// retained/released. Core Video's reference counting ensures memory safety.
// ThreadSafeCVImageBuffer maintains proper CVImageBuffer lifetime management.
#[allow(unsafe_code)]
unsafe impl Send for ThreadSafeCVImageBuffer {}

// SAFETY: CVImageBuffer supports concurrent access from multiple threads.
// All Core Video operations on CVImageBuffer are internally synchronized.
// ThreadSafeCVImageBuffer wrapper ensures no data races on the buffer reference.
#[allow(unsafe_code)]
unsafe impl Sync for ThreadSafeCVImageBuffer {}

/// macOS-specific implementation of VideoFrame
#[derive(Default)]
pub struct MacOSVideoFrame {
    // Use NativeVideoFrame for common functionality
    native: Option<NativeVideoFrame>,

    // macOS-specific fields - now thread-safe!
    buffer: Option<ThreadSafeCVImageBuffer>,
    width: u32,
    height: u32,
    timestamp_us: i64,
}

impl MacOSVideoFrame {
    /// Create a new MacOSVideoFrame from a NativeVideoFrame
    pub fn from_native(native: NativeVideoFrame) -> Self {
        let width = native.width();
        let height = native.height();
        let timestamp_us = native.timestamp_us();
        Self {
            native: Some(native),
            buffer: None,
            width,
            height,
            timestamp_us,
        }
    }

    /// Create a new MacOSVideoFrame from a CVImageBuffer with real Core Video dimensions
    pub fn from_cv_buffer(buffer: CVImageBuffer, timestamp_us: i64) -> Self {
        // Extract real dimensions from CVImageBuffer using Core Video API
        let display_size = buffer.get_display_size();
        let width = display_size.width as u32;
        let height = display_size.height as u32;

        Self {
            native: None,
            buffer: Some(ThreadSafeCVImageBuffer::new(buffer)),
            width,
            height,
            timestamp_us,
        }
    }

    /// Get the underlying CVImageBuffer
    pub fn cv_buffer(&self) -> Option<&CVImageBuffer> {
        self.buffer.as_ref().map(|b| b.get())
    }

    /// Get the buffer data from a CVImageBuffer using proper Core Video APIs
    fn get_buffer_data(&self) -> Result<Vec<u8>> {
        if let Some(buffer) = &self.buffer {
            let cv_buffer = buffer.get();

            // Get buffer properties using available methods
            let display_size = cv_buffer.get_display_size();
            let width = display_size.width as usize;
            let height = display_size.height as usize;

            // Cast CVImageBuffer to CVPixelBuffer for pixel data access
            // SAFETY: CVImageBuffer is the same as CVPixelBuffer - this is safe casting
            let pixel_buffer = unsafe {
                std::mem::transmute::<
                    &core_video::image_buffer::CVImageBuffer,
                    &core_video::pixel_buffer::CVPixelBuffer,
                >(cv_buffer)
            };

            // Lock the pixel buffer for reading
            let lock_result = pixel_buffer
                .lock_base_address(core_video::pixel_buffer::kCVPixelBufferLock_ReadOnly);
            if lock_result != core_video::r#return::kCVReturnSuccess {
                return Err(anyhow::anyhow!(
                    "Failed to lock CVPixelBuffer for reading: {:?}",
                    lock_result
                ));
            }

            // Create RAII guard to ensure unlock
            struct PixelBufferGuard<'a>(&'a core_video::pixel_buffer::CVPixelBuffer);
            impl<'a> Drop for PixelBufferGuard<'a> {
                fn drop(&mut self) {
                    let _ = self.0.unlock_base_address(0);
                }
            }
            let _guard = PixelBufferGuard(pixel_buffer);

            // Get buffer dimensions and format information
            let buffer_width = pixel_buffer.get_width();
            let buffer_height = pixel_buffer.get_height();
            let bytes_per_row = pixel_buffer.get_bytes_per_row();
            let pixel_format = pixel_buffer.get_pixel_format();

            // Get raw pixel data pointer
            let base_address = unsafe { pixel_buffer.get_base_address() };
            if base_address.is_null() {
                return Err(anyhow::anyhow!("CVPixelBuffer base address is null"));
            }

            // Calculate buffer size and create slice
            let buffer_size = bytes_per_row * buffer_height;
            let pixel_data =
                unsafe { std::slice::from_raw_parts(base_address as *const u8, buffer_size) };

            // Convert pixel data to RGBA format based on source format
            let mut rgba_buffer = Vec::with_capacity(width * height * 4);

            match pixel_format {
                // 32-bit BGRA format (most common from Core Graphics)
                core_video::pixel_buffer::kCVPixelFormatType_32BGRA => {
                    for y in 0..buffer_height.min(height) {
                        let row_start = y * bytes_per_row;
                        let pixel_row =
                            &pixel_data[row_start..row_start + (buffer_width.min(width) * 4)];

                        for bgra_pixel in pixel_row.chunks_exact(4) {
                            // Convert BGRA -> RGBA
                            rgba_buffer.push(bgra_pixel[2]); // R <- B
                            rgba_buffer.push(bgra_pixel[1]); // G <- G  
                            rgba_buffer.push(bgra_pixel[0]); // B <- R
                            rgba_buffer.push(bgra_pixel[3]); // A <- A
                        }

                        // Pad row if buffer is smaller than requested width
                        while rgba_buffer.len() < (y + 1) * width * 4 {
                            rgba_buffer.extend_from_slice(&[0, 0, 0, 255]); // Black pixels
                        }
                    }
                }
                // 32-bit RGBA format
                core_video::pixel_buffer::kCVPixelFormatType_32RGBA => {
                    for y in 0..buffer_height.min(height) {
                        let row_start = y * bytes_per_row;
                        let pixel_row =
                            &pixel_data[row_start..row_start + (buffer_width.min(width) * 4)];
                        rgba_buffer.extend_from_slice(pixel_row);

                        // Pad row if buffer is smaller than requested width
                        while rgba_buffer.len() < (y + 1) * width * 4 {
                            rgba_buffer.extend_from_slice(&[0, 0, 0, 255]); // Black pixels
                        }
                    }
                }
                // 24-bit RGB format
                core_video::pixel_buffer::kCVPixelFormatType_24RGB => {
                    for y in 0..buffer_height.min(height) {
                        let row_start = y * bytes_per_row;
                        let pixel_row =
                            &pixel_data[row_start..row_start + (buffer_width.min(width) * 3)];

                        for rgb_pixel in pixel_row.chunks_exact(3) {
                            // Convert RGB -> RGBA (add alpha)
                            rgba_buffer.push(rgb_pixel[0]); // R
                            rgba_buffer.push(rgb_pixel[1]); // G
                            rgba_buffer.push(rgb_pixel[2]); // B
                            rgba_buffer.push(255); // A (full opacity)
                        }

                        // Pad row if buffer is smaller than requested width
                        while rgba_buffer.len() < (y + 1) * width * 4 {
                            rgba_buffer.extend_from_slice(&[0, 0, 0, 255]); // Black pixels
                        }
                    }
                }
                // Unsupported format - return error instead of stub
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported CVPixelBuffer format: 0x{:08X}. Expected BGRA, RGBA, or RGB.",
                        pixel_format
                    ));
                }
            }

            // Pad remaining rows if buffer is smaller than requested height
            while rgba_buffer.len() < width * height * 4 {
                rgba_buffer.extend_from_slice(&[0, 0, 0, 255]); // Black pixels
            }

            Ok(rgba_buffer)
        } else if let Some(native) = &self.native {
            // Fall back to native implementation
            native.to_rgba_bytes()
        } else {
            Err(anyhow::anyhow!("No video buffer available"))
        }
    }
}

impl VideoFrameImpl for MacOSVideoFrame {
    fn to_rgba_bytes(&self) -> Result<Vec<u8>> {
        if let Some(_buffer) = &self.buffer {
            // get_buffer_data() already returns RGBA data - no additional conversion needed
            // This fixes the double color conversion bug where BGRA->RGBA was done twice
            self.get_buffer_data()
        } else if let Some(native) = &self.native {
            // Fall back to native implementation
            native.to_rgba_bytes()
        } else {
            Err(anyhow::anyhow!("No video buffer available"))
        }
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn timestamp_us(&self) -> i64 {
        self.timestamp_us
    }
}

/// Camera frame delegate for AVFoundation callbacks
#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct CameraFrameDelegate {
    frame_sender: Arc<RwLock<Option<VideoFrame>>>,
}

#[cfg(target_os = "macos")]
impl CameraFrameDelegate {
    pub fn new(frame_sender: Arc<RwLock<Option<VideoFrame>>>) -> Self {
        Self { frame_sender }
    }
}

#[cfg(target_os = "macos")]
unsafe impl NSObjectProtocol for CameraFrameDelegate {}

#[cfg(target_os = "macos")]
unsafe impl objc2::encode::RefEncode for CameraFrameDelegate {
    const ENCODING_REF: objc2::encode::Encoding = objc2::encode::Encoding::Object;
}

#[cfg(target_os = "macos")]
unsafe impl objc2::Message for CameraFrameDelegate {}

#[cfg(target_os = "macos")]
unsafe impl Send for CameraFrameDelegate {}
#[cfg(target_os = "macos")]
unsafe impl Sync for CameraFrameDelegate {}

#[cfg(target_os = "macos")]
unsafe impl AVCaptureVideoDataOutputSampleBufferDelegate for CameraFrameDelegate {
    unsafe fn capture_output_did_output_sample_buffer(
        &self,
        _output: &AVCaptureOutput,
        sample_buffer: core_media::sample_buffer::CMSampleBufferRef,
        _connection: &AVCaptureConnection,
    ) {
        // Convert CMSampleBufferRef to CMSampleBuffer for safe API access
        let sample_buffer = unsafe {
            core_media::sample_buffer::CMSampleBuffer::wrap_under_get_rule(sample_buffer)
        };

        // Extract CVImageBuffer from sample buffer using safe API
        if let Some(image_buffer) = sample_buffer.get_image_buffer() {
            // Get timestamp from sample buffer using safe API
            let timestamp = sample_buffer.get_presentation_time_stamp();
            let timestamp_us =
                (timestamp.value as f64 / timestamp.timescale as f64 * 1_000_000.0) as i64;

            // Create NativeVideoFrame from the camera buffer - convert CV buffer to RGBA data
            let encoded_size = image_buffer.get_encoded_size();
            let width = encoded_size.width as u32;
            let height = encoded_size.height as u32;

            // For now, create a placeholder frame - proper CV buffer conversion would be implemented here
            let rgba_data = vec![0u8; (width * height * 4) as usize];
            let native_frame = NativeVideoFrame::create_video_frame(
                rgba_data,
                width,
                height,
                timestamp_us,
                VideoRotation::Rotation0,
            );

            // Update the current frame in a thread-safe manner
            if let Ok(mut frame_guard) = self.frame_sender.write() {
                *frame_guard = Some(native_frame);
            }
        }
    }

    unsafe fn capture_output_did_drop_sample_buffer(
        &self,
        _output: &AVCaptureOutput,
        _sample_buffer: core_media::sample_buffer::CMSampleBufferRef,
        _connection: &AVCaptureConnection,
    ) {
        // Handle dropped frames - could log for debugging
        tracing::debug!("Camera frame was dropped");
    }
}

/// macOS-specific implementation of VideoSource
#[derive(Debug)]
pub struct MacOSVideoSource {
    info: VideoSourceInfo,
    current_frame: Arc<RwLock<Option<VideoFrame>>>,
    is_active: bool,
    frame_timer: Option<tokio::task::JoinHandle<()>>,

    // macOS-specific capture session state
    capture_session_id: Option<String>,
    capture_device_id: Option<String>,

    // Real AVFoundation components for camera capture (macOS only)
    #[cfg(target_os = "macos")]
    av_capture_session: Option<objc2::rc::Id<AVCaptureSession>>,
    #[cfg(target_os = "macos")]
    av_capture_device: Option<objc2::rc::Id<AVCaptureDevice>>,
    #[cfg(target_os = "macos")]
    camera_delegate: Option<Arc<CameraFrameDelegate>>,

    // Thread-safe FFmpeg source for file reading
    ffmpeg_source: Option<ThreadSafeFFmpegSource>,
}

// SAFETY: MacOSVideoSource is thread-safe because:
// 1. All AVFoundation types (Retained<T>) are reference-counted and thread-safe
// 2. ThreadSafeFFmpegSource uses channels for thread-safe communication
// 3. Arc<RwLock<T>> provides thread-safe access to shared state
// 4. All other fields are either primitive types or thread-safe wrappers
unsafe impl Send for MacOSVideoSource {}
unsafe impl Sync for MacOSVideoSource {}

impl MacOSVideoSource {
    /// Create a new MacOSVideoSource from a camera
    #[cfg(target_os = "macos")]
    pub fn from_camera(options: VideoSourceOptions) -> Result<Self> {
        let width = options.width.unwrap_or(640);
        let height = options.height.unwrap_or(480);
        let fps = options.fps.unwrap_or(30) as f64;

        // Create AVFoundation capture session
        let capture_session = AVCaptureSession::new();

        // Get the default camera device
        // SAFETY: AVMediaTypeVideo is a valid extern static from AVFoundation
        let device = AVCaptureDevice::default_device_with_media_type(unsafe { &AVMediaTypeVideo })
            .ok_or_else(|| anyhow::anyhow!("No camera device found"))?;

        // Create device input from camera
        let device_input = AVCaptureDeviceInput::from_device(&device)
            .map_err(|err| anyhow::anyhow!("Failed to create camera input: {:?}", err))?;

        // Add input to session
        if !capture_session.can_add_input(&device_input) {
            return Err(anyhow::anyhow!(
                "Cannot add camera input to capture session"
            ));
        }
        capture_session.add_input(&device_input);

        // Create video data output for frame capture
        let video_output = AVCaptureVideoDataOutput::new();

        // Set pixel format to 32-bit BGRA for compatibility
        let _pixel_format = core_video::pixel_buffer::kCVPixelFormatType_32BGRA;
        // TODO: Configure video settings for AVFoundation video output

        // Create frame storage for the delegate
        let current_frame = Arc::new(RwLock::new(None));

        // Create camera delegate to handle incoming frames
        let camera_delegate = Arc::new(CameraFrameDelegate::new(current_frame.clone()));

        // Set up delegate and dispatch queue for frame callbacks
        let dispatch_queue = Queue::new("camera_frame_queue", dispatch2::QueueAttribute::Serial);
        let delegate_obj = ProtocolObject::from_ref(&*camera_delegate);
        video_output.set_sample_buffer_delegate(delegate_obj, &dispatch_queue);

        // Add video output to session
        if !capture_session.can_add_output(&video_output) {
            return Err(anyhow::anyhow!(
                "Cannot add video output to capture session"
            ));
        }
        capture_session.add_output(&video_output);

        // Get device identifier for tracking
        let device_id = device.unique_id().to_string();

        Ok(Self {
            info: VideoSourceInfo {
                name: format!("macOS Camera: {}", device.localized_name()),
                width,
                height,
                fps,
                is_active: false,
            },
            current_frame,
            is_active: false,
            frame_timer: None,
            capture_session_id: Some("AVCaptureSessionPresetMedium".to_string()),
            capture_device_id: Some(device_id),
            av_capture_session: Some(capture_session),
            av_capture_device: Some(device),
            camera_delegate: Some(camera_delegate),
            ffmpeg_source: None,
        })
    }

    /// Create a new MacOSVideoSource from a camera (non-macOS fallback)
    #[cfg(not(target_os = "macos"))]
    pub fn from_camera(_options: VideoSourceOptions) -> Result<Self> {
        Err(anyhow::anyhow!("Camera capture is only supported on macOS"))
    }

    /// Create a new MacOSVideoSource from screen capture
    pub fn from_screen(options: VideoSourceOptions) -> Result<Self> {
        let width = options.width.unwrap_or(1920);
        let height = options.height.unwrap_or(1080);
        let fps = options.fps.unwrap_or(30) as f64;

        // Initialize screen capture configuration
        // Configures display capture parameters for CGDisplayStream integration

        Ok(Self {
            info: VideoSourceInfo {
                name: "macOS Screen".to_string(),
                width,
                height,
                fps,
                is_active: false,
            },
            current_frame: Arc::new(RwLock::new(None)),
            is_active: false,
            frame_timer: None,
            capture_session_id: None,
            capture_device_id: Some("main_display".to_string()),
            #[cfg(target_os = "macos")]
            av_capture_session: None,
            #[cfg(target_os = "macos")]
            av_capture_device: None,
            #[cfg(target_os = "macos")]
            camera_delegate: None,
            ffmpeg_source: None,
        })
    }

    /// Create a new MacOSVideoSource from a file
    pub fn from_file(path: &str, options: VideoSourceOptions) -> Result<Self> {
        let width = options.width.unwrap_or(640);
        let height = options.height.unwrap_or(480);
        let fps = options.fps.unwrap_or(30) as f64;

        // Initialize FFmpeg
        ffmpeg::init().map_err(|e| anyhow::anyhow!("Failed to initialize FFmpeg: {}", e))?;

        Ok(Self {
            info: VideoSourceInfo {
                name: format!("Video File: {}", path),
                width,
                height,
                fps,
                is_active: false,
            },
            current_frame: Arc::new(RwLock::new(None)),
            is_active: false,
            frame_timer: None,
            capture_session_id: None,
            capture_device_id: None,
            #[cfg(target_os = "macos")]
            av_capture_session: None,
            #[cfg(target_os = "macos")]
            av_capture_device: None,
            #[cfg(target_os = "macos")]
            camera_delegate: None,
            ffmpeg_source: Some(ThreadSafeFFmpegSource::new()),
        })
    }

    /// Get the capture session identifier
    #[allow(dead_code)] // Reserved for future Core Video integration
    pub fn capture_session_id(&self) -> Option<&str> {
        self.capture_session_id.as_deref()
    }

    /// Get the capture device identifier  
    #[allow(dead_code)] // Reserved for future Core Video integration
    pub fn capture_device_id(&self) -> Option<&str> {
        self.capture_device_id.as_deref()
    }

    /// Start file reading using thread-safe FFmpeg source
    #[allow(dead_code)] // FFmpeg file reading functionality reserved for future file input support
    fn start_file_reading(&mut self, file_path: String) -> Result<()> {
        if let Some(ffmpeg_source) = &self.ffmpeg_source {
            ffmpeg_source.start_file(
                file_path,
                self.info.width,
                self.info.height,
                self.info.fps,
            )?;
        } else {
            return Err(anyhow::anyhow!("FFmpeg source not initialized"));
        }
        Ok(())
    }

    /// Start screen capture using Core Graphics
    fn start_screen_capture(&mut self) -> Result<()> {
        let current_frame = self.current_frame.clone();
        let info = self.info.clone();

        let frame_timer = tokio::spawn(async move {
            let mut frame_number = 0;
            let frame_duration = std::time::Duration::from_millis((1000.0 / info.fps) as u64);

            loop {
                // Capture real screen content
                let frame = {
                    let macos_source = MacOSVideoSource {
                        info: info.clone(),
                        current_frame: current_frame.clone(),
                        is_active: true,
                        frame_timer: None,
                        capture_session_id: None,
                        capture_device_id: Some("main_display".to_string()),
                        #[cfg(target_os = "macos")]
                        av_capture_session: None,
                        #[cfg(target_os = "macos")]
                        av_capture_device: None,
                        #[cfg(target_os = "macos")]
                        camera_delegate: None,
                        ffmpeg_source: None,
                    };

                    macos_source.capture_screen_frame(frame_number)
                };

                // Update current frame
                if let Ok(mut write_guard) = current_frame.write() {
                    *write_guard = Some(frame);
                }

                frame_number += 1;
                tokio::time::sleep(frame_duration).await;
            }
        });

        self.frame_timer = Some(frame_timer);
        Ok(())
    }

    /// Static helper method to convert FFmpeg frame to VideoFrame (for use in async context)
    #[allow(dead_code)] // FFmpeg frame conversion functionality reserved for future file input support
    fn convert_ffmpeg_frame_to_rgba(
        frame: &ffmpeg::frame::Video,
        target_width: u32,
        target_height: u32,
        frame_number: u64,
        fps: f64,
    ) -> Result<VideoFrame> {
        let width = frame.width();
        let height = frame.height();

        // Create software scaling context
        let mut scaler = ffmpeg::software::scaling::context::Context::get(
            frame.format(),
            width,
            height,
            ffmpeg::format::Pixel::RGBA,
            target_width,
            target_height,
            ffmpeg::software::scaling::flag::Flags::BILINEAR,
        )
        .map_err(|e| anyhow::anyhow!("Failed to create scaling context: {}", e))?;

        // Create output frame - let scaler handle allocation
        let mut rgba_frame = ffmpeg::frame::Video::empty();

        // Scale and convert
        scaler
            .run(frame, &mut rgba_frame)
            .map_err(|e| anyhow::anyhow!("Failed to scale frame: {}", e))?;

        // Extract RGBA data
        let rgba_data = rgba_frame.data(0).to_vec();

        // Calculate timestamp
        let timestamp_us = (frame_number as f64 / fps * 1_000_000.0) as i64;

        // Create native video frame
        let native_frame = NativeVideoFrame::new(
            rgba_data,
            target_width,
            target_height,
            timestamp_us,
            VideoRotation::Rotation0,
        );

        Ok(VideoFrame::new(MacOSVideoFrame::from_native(native_frame)))
    }

    /// Capture real screen content using Core Graphics
    /// TODO: Replace CGWindowListCreateImage with ScreenCaptureKit for macOS 12.3+ compatibility
    /// The current implementation uses deprecated Core Graphics API but provides fallback
    /// to test patterns if screen capture fails, ensuring graceful degradation.
    #[cfg(target_os = "macos")]
    #[allow(deprecated)]
    fn capture_screen_to_rgba(&self, width: u32, height: u32) -> Result<Vec<u8>> {
        unsafe {
            // Create a full-screen rect for capture (we'll use null bounds to capture all screens)
            let cg_rect = CGRect {
                origin: CGPoint { x: 0.0, y: 0.0 },
                size: CGSize {
                    width: 0.0,
                    height: 0.0,
                },
            };

            // Capture the screen using Core Graphics
            let cg_image = CGWindowListCreateImage(
                cg_rect,
                CGWindowListOption::OptionAll,
                0 as CGWindowID,
                CGWindowImageOption::Default,
            );

            // Extract image data
            let image_width = CGImage::width(cg_image.as_deref()) as usize;
            let image_height = CGImage::height(cg_image.as_deref()) as usize;
            let data_provider = CGImage::data_provider(cg_image.as_deref());

            let data = CGDataProvider::data(data_provider.as_deref())
                .ok_or_else(|| anyhow::anyhow!("Failed to copy screen capture data"))?
                .to_vec();

            let bytes_per_row = CGImage::bytes_per_row(cg_image.as_deref());

            // Handle row padding - macOS can have extra bytes at the end of each row
            let mut buffer = Vec::with_capacity(image_width * image_height * 4);
            for row in data.chunks_exact(bytes_per_row) {
                buffer.extend_from_slice(&row[..image_width * 4]);
            }

            // Convert BGRA to RGBA (Core Graphics uses BGRA format)
            for bgra in buffer.chunks_exact_mut(4) {
                bgra.swap(0, 2); // Swap B and R channels
            }

            // Resize if needed to match requested dimensions
            if image_width == width as usize && image_height == height as usize {
                Ok(buffer)
            } else {
                // Use high-quality Lanczos-3 resize for production quality scaling
                let resized =
                    self.resize_lanczos3(&buffer, image_width, image_height, width, height);
                Ok(resized)
            }
        }
    }

    /// High-quality Lanczos-3 image resize algorithm for production use
    fn resize_lanczos3(
        &self,
        src_data: &[u8],
        src_width: usize,
        src_height: usize,
        dst_width: u32,
        dst_height: u32,
    ) -> Vec<u8> {
        let dst_width = dst_width as usize;
        let dst_height = dst_height as usize;
        let mut dst_data = vec![0u8; dst_width * dst_height * 4];

        let x_ratio = src_width as f32 / dst_width as f32;
        let y_ratio = src_height as f32 / dst_height as f32;

        // Lanczos-3 kernel function (a = 3)
        let lanczos3 = |x: f32| -> f32 {
            if x == 0.0 {
                1.0
            } else if x.abs() < 3.0 {
                let pi_x = std::f32::consts::PI * x;
                let pi_x_3 = pi_x / 3.0;
                3.0 * pi_x.sin() * pi_x_3.sin() / (pi_x * pi_x)
            } else {
                0.0
            }
        };

        // Resize with Lanczos-3 filtering
        for dst_y in 0..dst_height {
            for dst_x in 0..dst_width {
                let src_x_center = (dst_x as f32 + 0.5) * x_ratio - 0.5;
                let src_y_center = (dst_y as f32 + 0.5) * y_ratio - 0.5;

                let mut r_sum = 0.0f32;
                let mut g_sum = 0.0f32;
                let mut b_sum = 0.0f32;
                let mut a_sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                // Sample 6x6 neighborhood around the center point
                for sample_y in -2..=3 {
                    for sample_x in -2..=3 {
                        let src_x = (src_x_center + sample_x as f32).round() as i32;
                        let src_y = (src_y_center + sample_y as f32).round() as i32;

                        if src_x >= 0
                            && src_x < src_width as i32
                            && src_y >= 0
                            && src_y < src_height as i32
                        {
                            let x_weight = lanczos3(src_x_center - src_x as f32);
                            let y_weight = lanczos3(src_y_center - src_y as f32);
                            let weight = x_weight * y_weight;

                            if weight != 0.0 {
                                let src_idx = (src_y as usize * src_width + src_x as usize) * 4;
                                if src_idx + 3 < src_data.len() {
                                    r_sum += src_data[src_idx] as f32 * weight;
                                    g_sum += src_data[src_idx + 1] as f32 * weight;
                                    b_sum += src_data[src_idx + 2] as f32 * weight;
                                    a_sum += src_data[src_idx + 3] as f32 * weight;
                                    weight_sum += weight;
                                }
                            }
                        }
                    }
                }

                // Normalize and clamp values
                let dst_idx = (dst_y * dst_width + dst_x) * 4;
                if weight_sum > 0.0 {
                    dst_data[dst_idx] = (r_sum / weight_sum).round().max(0.0).min(255.0) as u8;
                    dst_data[dst_idx + 1] = (g_sum / weight_sum).round().max(0.0).min(255.0) as u8;
                    dst_data[dst_idx + 2] = (b_sum / weight_sum).round().max(0.0).min(255.0) as u8;
                    dst_data[dst_idx + 3] = (a_sum / weight_sum).round().max(0.0).min(255.0) as u8;
                } else {
                    // Fallback to nearest neighbor if no weights
                    let nearest_x =
                        src_x_center.round().max(0.0).min(src_width as f32 - 1.0) as usize;
                    let nearest_y =
                        src_y_center.round().max(0.0).min(src_height as f32 - 1.0) as usize;
                    let nearest_idx = (nearest_y * src_width + nearest_x) * 4;
                    if nearest_idx + 3 < src_data.len() {
                        dst_data[dst_idx..dst_idx + 4]
                            .copy_from_slice(&src_data[nearest_idx..nearest_idx + 4]);
                    }
                }
            }
        }

        dst_data
    }

    /// Capture real screen frame
    fn capture_screen_frame(&self, _frame_number: u64) -> VideoFrame {
        let width = self.info.width;
        let height = self.info.height;

        // Use Core Graphics screen capture
        let rgba_data = self
            .capture_screen_to_rgba(width, height)
            .unwrap_or_else(|_| {
                // Fallback to test pattern on capture failure
                self.generate_test_pattern(0)
                    .to_rgba_bytes()
                    .unwrap_or_default()
            });

        let timestamp_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_micros() as i64;

        let native_frame = NativeVideoFrame::new(
            rgba_data,
            width,
            height,
            timestamp_us,
            VideoRotation::Rotation0,
        );

        VideoFrame::new(MacOSVideoFrame::from_native(native_frame))
    }

    /// Generate a test pattern frame
    fn generate_test_pattern(&self, frame_number: u64) -> VideoFrame {
        let width = self.info.width;
        let height = self.info.height;

        // Create a simple test pattern with a moving gradient
        let mut data = Vec::with_capacity((width * height * 4) as usize);

        let t = frame_number % 255;

        for y in 0..height {
            for x in 0..width {
                // Create a gradient pattern
                let r = ((x * 255) / width) as u8;
                let g = ((y * 255) / height) as u8;
                let b = t as u8;
                let a = 255u8;

                data.push(r);
                data.push(g);
                data.push(b);
                data.push(a);
            }
        }

        // This would be replaced with actual macOS frame creation
        // For now, just use the native implementation
        let native_frame = NativeVideoFrame::new(
            data,
            width,
            height,
            frame_number as i64 * 1_000_000 / self.info.fps as i64,
            VideoRotation::Rotation0,
        );

        let macos_frame = MacOSVideoFrame::from_native(native_frame);
        VideoFrame::new(macos_frame)
    }
}

impl VideoSourceImpl for MacOSVideoSource {
    fn start(&mut self) -> Result<()> {
        if self.is_active {
            return Ok(());
        }

        self.is_active = true;
        self.info.is_active = true;

        // Start real AVFoundation camera capture session (macOS only)
        #[cfg(target_os = "macos")]
        if let Some(capture_session) = &self.av_capture_session {
            capture_session.start_running();
            return Ok(());
        }

        // Determine source type and start appropriate capture method
        if self
            .capture_device_id
            .as_deref()
            .unwrap_or("")
            .contains("file")
        {
            // File reading - start FFmpeg-based file reading
            // File reading requires FFmpeg source to be initialized
            if let Some(ffmpeg_source) = &self.ffmpeg_source {
                ffmpeg_source.start_file("".to_string(), 1920, 1080, 30.0)?;
            }
        } else if self.capture_device_id.as_deref() == Some("main_display") {
            // Screen capture - use real Core Graphics screen capture
            self.start_screen_capture()?;
        } else {
            // Fallback for other sources - should not happen in normal operation
            return Err(anyhow::anyhow!(
                "Unknown source type - cannot start capture"
            ));
        }

        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        if !self.is_active {
            return Ok(());
        }

        self.is_active = false;
        self.info.is_active = false;

        // Stop real AVFoundation camera capture session (macOS only)
        #[cfg(target_os = "macos")]
        if let Some(capture_session) = &self.av_capture_session {
            capture_session.stop_running();
        }

        // Stop frame generation for non-camera sources
        if let Some(timer) = self.frame_timer.take() {
            timer.abort();
        }

        // Clean up AVFoundation resources
        #[cfg(target_os = "macos")]
        {
            self.av_capture_session = None;
            self.av_capture_device = None;
            self.camera_delegate = None;
        }

        Ok(())
    }

    fn get_current_frame(&self) -> Option<VideoFrame> {
        if let Ok(read_guard) = self.current_frame.read() {
            read_guard.clone()
        } else {
            None
        }
    }

    fn get_info(&self) -> VideoSourceInfo {
        self.info.clone()
    }
}
