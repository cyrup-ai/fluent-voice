use crate::video_frame::{VideoFrame, VideoFrameImpl};
use anyhow::Result;
use std::fmt;

/// Native VideoBuffer implementation - simplified for cross-platform compatibility
pub struct NativeVideoBuffer {
    buffer: Vec<u8>,
    width: u32,
    height: u32,
}

impl Clone for NativeVideoBuffer {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            width: self.width,
            height: self.height,
        }
    }
}

impl NativeVideoBuffer {
    /// Create a new NativeVideoBuffer from raw RGBA data
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
        // Assume the buffer is already in RGBA format or create a placeholder
        if self.buffer.len() == (self.width * self.height * 4) as usize {
            Ok(self.buffer.clone())
        } else {
            // Create a gray placeholder if buffer size doesn't match expected RGBA size
            Ok(vec![128u8; (self.width * self.height * 4) as usize])
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
    #[allow(dead_code)]
    Rotation90,
    /// 180 degrees rotation
    #[allow(dead_code)]
    Rotation180,
    /// 270 degrees rotation
    #[allow(dead_code)]
    Rotation270,
}

impl NativeVideoFrame {
    /// Create a new NativeVideoFrame from raw RGBA data
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
    #[allow(dead_code)]
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
