use crate::VideoFrame;
use anyhow::Result;
use futures::Stream;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[cfg(target_os = "macos")]
use crate::macos::MacOSVideoSource;

#[cfg(not(target_os = "macos"))]
use crate::generic::GenericVideoSource;

/// Represents a source of video frames, such as a camera, screen capture, or file
#[derive(Debug, Clone)]
pub struct VideoSource {
    #[cfg(target_os = "macos")]
    inner: Arc<Mutex<MacOSVideoSource>>,

    #[cfg(not(target_os = "macos"))]
    inner: Arc<Mutex<GenericVideoSource>>,
}

/// Configuration options for a video source
#[derive(Debug, Clone, Default)]
pub struct VideoSourceOptions {
    /// The desired width of the video source
    pub width: Option<u32>,

    /// The desired height of the video source
    pub height: Option<u32>,

    /// The desired frame rate of the video source
    pub fps: Option<u32>,
}

impl VideoSource {
    /// Create a new video source from a camera
    pub fn from_camera(options: VideoSourceOptions) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let inner = MacOSVideoSource::from_camera(options)?;
            Ok(Self {
                inner: Arc::new(Mutex::new(inner)),
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            let inner = GenericVideoSource::from_camera(options)?;
            Ok(Self {
                inner: Arc::new(Mutex::new(inner)),
            })
        }
    }

    /// Create a new video source from screen capture
    pub fn from_screen(options: VideoSourceOptions) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let inner = MacOSVideoSource::from_screen(options)?;
            Ok(Self {
                inner: Arc::new(Mutex::new(inner)),
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            let inner = GenericVideoSource::from_screen(options)?;
            Ok(Self {
                inner: Arc::new(Mutex::new(inner)),
            })
        }
    }

    /// Create a new video source from a file
    pub fn from_file(path: &str, options: VideoSourceOptions) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let inner = MacOSVideoSource::from_file(path, options)?;
            Ok(Self {
                inner: Arc::new(Mutex::new(inner)),
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            let inner = GenericVideoSource::from_file(path, options)?;
            Ok(Self {
                inner: Arc::new(Mutex::new(inner)),
            })
        }
    }

    /// Start capturing video frames
    pub fn start(&self) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire video source lock for start"))?;
        inner.start()
    }

    /// Stop capturing video frames
    pub fn stop(&self) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire video source lock for stop"))?;
        inner.stop()
    }

    /// Get the current frame from the video source
    pub fn get_current_frame(&self) -> Option<VideoFrame> {
        let inner = self.inner.lock().ok()?;
        inner.get_current_frame()
    }

    /// Get a stream of video frames from the video source
    pub fn get_frame_stream(&self) -> impl Stream<Item = VideoFrame> + Send + 'static {
        let inner = self.inner.clone();
        async_stream::stream! {
            loop {
                if let Some(frame) = {
                    match inner.lock() {
                        Ok(inner) => inner.get_current_frame(),
                        Err(_) => None, // Failed to acquire lock, skip this frame
                    }
                } {
                    yield frame;
                }
                tokio::time::sleep(Duration::from_millis(33)).await; // ~30fps (matches capture rate)
            }
        }
    }

    /// Get the width of the video source
    pub fn width(&self) -> u32 {
        let inner = match self.inner.lock() {
            Ok(inner) => inner,
            Err(_) => return 0, // Default to 0 if lock fails
        };
        let info = inner.get_info();
        info.width
    }

    /// Get the height of the video source
    pub fn height(&self) -> u32 {
        let inner = match self.inner.lock() {
            Ok(inner) => inner,
            Err(_) => return 0, // Default to 0 if lock fails
        };
        let info = inner.get_info();
        info.height
    }
}

/// Trait for platform-specific video source implementations
pub(crate) trait VideoSourceImpl: Send + Sync {
    /// Start capturing video frames
    fn start(&mut self) -> Result<()>;

    /// Stop capturing video frames
    fn stop(&mut self) -> Result<()>;

    /// Get the current frame from the video source
    fn get_current_frame(&self) -> Option<VideoFrame>;

    /// Get information about the video source
    #[allow(dead_code)]
    fn get_info(&self) -> VideoSourceInfo;
}

/// Information about a video source
#[derive(Debug, Clone)]
pub struct VideoSourceInfo {
    /// The name of the video source
    #[allow(dead_code)]
    pub name: String,

    /// The width of the video source
    pub width: u32,

    /// The height of the video source
    pub height: u32,

    /// The frame rate of the video source
    pub fps: f64,

    /// Whether the video source is currently active
    pub is_active: bool,
}
