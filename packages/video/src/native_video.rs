use crate::video_frame::{VideoFrame, VideoFrameImpl};
use anyhow::Result;
use std::fmt;

/// Native VideoBuffer implementation from libwebrtc
#[derive(Clone)]
pub struct NativeVideoBuffer {
    #[cfg(not(all(target_os = "windows", target_env = "gnu")))]
    buffer: Box<dyn libwebrtc::video_frame::VideoBuffer>,

    #[cfg(all(target_os = "windows", target_env = "gnu"))]
    buffer: Vec<u8>,

    width: u32,
    height: u32,
}

impl NativeVideoBuffer {
    /// Create a new NativeVideoBuffer from libwebrtc VideoFrameBuffer
    #[cfg(not(all(target_os = "windows", target_env = "gnu")))]
    pub fn new(buffer: Box<dyn libwebrtc::video_frame::VideoBuffer>) -> Self {
        let width = buffer.width();
        let height = buffer.height();
        Self {
            buffer,
            width,
            height,
        }
    }

    /// Create a new NativeVideoBuffer from raw data (for platforms without libwebrtc)
    #[cfg(all(target_os = "windows", target_env = "gnu"))]
    pub fn new(buffer: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            buffer,
            width,
            height,
        }
    }

    /// Get the width of the video buffer
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the height of the video buffer
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Convert to RGBA bytes for rendering
    pub fn to_rgba_bytes(&self) -> Result<Vec<u8>> {
        #[cfg(not(all(target_os = "windows", target_env = "gnu")))]
        {
            // This is a simplified implementation - actual implementation would depend
            // on the specific type of VideoFrameBuffer (I420, NV12, ARGB, etc.)
            let buffer_type = self.buffer.type_();
            match buffer_type {
                libwebrtc::video_frame::VideoBufferType::kI420 => {
                    let i420 = self.buffer.to_i420();
                    let mut rgba_buffer = vec![0u8; (self.width * self.height * 4) as usize];

                    // Convert I420 to RGBA
                    // This would need a proper YUV to RGB conversion algorithm
                    // Simplified implementation for demonstration
                    let y_data = i420.data_y();
                    let u_data = i420.data_u();
                    let v_data = i420.data_v();

                    let y_stride = i420.stride_y() as usize;
                    let u_stride = i420.stride_u() as usize;
                    let v_stride = i420.stride_v() as usize;

                    for y in 0..self.height as usize {
                        for x in 0..self.width as usize {
                            let y_index = y * y_stride + x;
                            let u_index = (y / 2) * u_stride + (x / 2);
                            let v_index = (y / 2) * v_stride + (x / 2);

                            let y_value = y_data[y_index] as f32;
                            let u_value = u_data[u_index] as f32 - 128.0;
                            let v_value = v_data[v_index] as f32 - 128.0;

                            // YUV to RGB conversion
                            let r = (y_value + 1.402 * v_value).clamp(0.0, 255.0) as u8;
                            let g = (y_value - 0.344 * u_value - 0.714 * v_value).clamp(0.0, 255.0)
                                as u8;
                            let b = (y_value + 1.772 * u_value).clamp(0.0, 255.0) as u8;

                            let rgba_index = (y * self.width as usize + x) * 4;
                            rgba_buffer[rgba_index] = r;
                            rgba_buffer[rgba_index + 1] = g;
                            rgba_buffer[rgba_index + 2] = b;
                            rgba_buffer[rgba_index + 3] = 255; // Alpha
                        }
                    }

                    Ok(rgba_buffer)
                }
                libwebrtc::video_frame::VideoBufferType::kARGB => {
                    // For ARGB, we can directly access the data
                    let argb = self.buffer.to_argb();
                    let argb_data = argb.data();

                    // Convert ARGB to RGBA
                    let mut rgba_buffer = vec![0u8; (self.width * self.height * 4) as usize];
                    for i in 0..self.width as usize * self.height as usize {
                        let argb_index = i * 4;
                        let rgba_index = i * 4;

                        rgba_buffer[rgba_index] = argb_data[argb_index + 1]; // R
                        rgba_buffer[rgba_index + 1] = argb_data[argb_index + 2]; // G
                        rgba_buffer[rgba_index + 2] = argb_data[argb_index + 3]; // B
                        rgba_buffer[rgba_index + 3] = argb_data[argb_index]; // A
                    }

                    Ok(rgba_buffer)
                }
                _ => {
                    // For other formats, we would need to implement specific conversions
                    // For now, just return a placeholder
                    let rgba_buffer = vec![128u8; (self.width * self.height * 4) as usize];
                    Ok(rgba_buffer)
                }
            }
        }

        #[cfg(all(target_os = "windows", target_env = "gnu"))]
        {
            // For platforms without libwebrtc, we assume the buffer is already in RGBA format
            Ok(self.buffer.clone())
        }
    }
}

impl fmt::Debug for NativeVideoBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeVideoBuffer")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

/// Native VideoFrame implementation
#[derive(Clone)]
pub struct NativeVideoFrame {
    buffer: NativeVideoBuffer,
    timestamp_us: i64,
    rotation: VideoRotation,
}

/// Video rotation values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoRotation {
    /// No rotation (0 degrees)
    Rotation0,
    /// 90 degrees rotation
    Rotation90,
    /// 180 degrees rotation
    Rotation180,
    /// 270 degrees rotation
    Rotation270,
}

impl NativeVideoFrame {
    /// Create a new NativeVideoFrame
    #[cfg(not(all(target_os = "windows", target_env = "gnu")))]
    pub fn new(
        buffer: Box<dyn libwebrtc::video_frame::VideoBuffer>,
        timestamp_us: i64,
        rotation: VideoRotation,
    ) -> Self {
        Self {
            buffer: NativeVideoBuffer::new(buffer),
            timestamp_us,
            rotation,
        }
    }

    /// Create a new NativeVideoFrame from raw data (for platforms without libwebrtc)
    #[cfg(all(target_os = "windows", target_env = "gnu"))]
    pub fn new(
        buffer: Vec<u8>,
        width: u32,
        height: u32,
        timestamp_us: i64,
        rotation: VideoRotation,
    ) -> Self {
        Self {
            buffer: NativeVideoBuffer::new(buffer, width, height),
            timestamp_us,
            rotation,
        }
    }

    /// Create a VideoFrame from this NativeVideoFrame
    pub fn to_video_frame(&self) -> VideoFrame {
        #[cfg(target_os = "macos")]
        {
            use crate::macos::MacOSVideoFrame;
            let frame = MacOSVideoFrame::from_native(self.clone());
            VideoFrame::new(frame)
        }

        #[cfg(not(target_os = "macos"))]
        {
            use crate::generic::GenericVideoFrame;
            let frame = GenericVideoFrame::from_native(self.clone());
            VideoFrame::new(frame)
        }
    }
}

impl fmt::Debug for NativeVideoFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeVideoFrame")
            .field("buffer", &self.buffer)
            .field("timestamp_us", &self.timestamp_us)
            .field("rotation", &self.rotation)
            .finish()
    }
}

impl VideoFrameImpl for NativeVideoFrame {
    fn to_rgba_bytes(&self) -> Result<Vec<u8>> {
        self.buffer.to_rgba_bytes()
    }

    fn width(&self) -> u32 {
        self.buffer.width()
    }

    fn height(&self) -> u32 {
        self.buffer.height()
    }

    fn timestamp_us(&self) -> i64 {
        self.timestamp_us
    }
}
