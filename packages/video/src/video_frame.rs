use anyhow::Result;
use std::fmt;
use std::sync::Arc;

#[cfg(target_os = "macos")]
use crate::macos::MacOSVideoFrame;

#[cfg(not(target_os = "macos"))]
use crate::generic::GenericVideoFrame;

/// Represents a video frame from any source
#[derive(Clone)]
pub struct VideoFrame {
    #[cfg(target_os = "macos")]
    inner: Arc<MacOSVideoFrame>,

    #[cfg(not(target_os = "macos"))]
    inner: Arc<GenericVideoFrame>,
}

impl VideoFrame {
    /// Create a new video frame from platform-specific implementation
    #[cfg(target_os = "macos")]
    pub(crate) fn new(inner: MacOSVideoFrame) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Create a new video frame from platform-specific implementation
    #[cfg(not(target_os = "macos"))]
    pub(crate) fn new(inner: GenericVideoFrame) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Convert frame data to RGBA bytes for rendering
    pub fn to_rgba_bytes(&self) -> Result<Vec<u8>> {
        self.inner.to_rgba_bytes()
    }

    /// Get the width of the video frame
    pub fn width(&self) -> u32 {
        self.inner.width()
    }

    /// Get the height of the video frame
    pub fn height(&self) -> u32 {
        self.inner.height()
    }

    /// Get the timestamp of the video frame
    pub fn timestamp_us(&self) -> i64 {
        self.inner.timestamp_us()
    }

    /// Check if the video frame is empty
    pub fn is_empty(&self) -> bool {
        self.width() == 0 || self.height() == 0
    }

    /// Get the underlying CVImageBuffer (macOS only)
    #[cfg(target_os = "macos")]
    pub fn cv_buffer(&self) -> Option<&core_video::image_buffer::CVImageBuffer> {
        self.inner.cv_buffer()
    }
}

impl Default for VideoFrame {
    fn default() -> Self {
        #[cfg(target_os = "macos")]
        {
            use crate::macos::MacOSVideoFrame;
            Self::new(MacOSVideoFrame::default())
        }

        #[cfg(not(target_os = "macos"))]
        {
            use crate::generic::GenericVideoFrame;
            Self::new(GenericVideoFrame::default())
        }
    }
}

impl fmt::Debug for VideoFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VideoFrame")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("timestamp_us", &self.timestamp_us())
            .finish()
    }
}

/// Trait for platform-specific video frame implementations
pub(crate) trait VideoFrameImpl: Send + Sync {
    /// Convert frame data to RGBA bytes for rendering
    fn to_rgba_bytes(&self) -> Result<Vec<u8>>;

    /// Get the width of the video frame
    fn width(&self) -> u32;

    /// Get the height of the video frame
    fn height(&self) -> u32;

    /// Get the timestamp of the video frame in microseconds
    fn timestamp_us(&self) -> i64;
}
