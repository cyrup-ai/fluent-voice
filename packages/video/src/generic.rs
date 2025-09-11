use crate::native_video::{NativeVideoFrame, VideoRotation};
use crate::video_frame::{VideoFrame, VideoFrameImpl};
use crate::video_source::{VideoSourceImpl, VideoSourceInfo, VideoSourceOptions};
use anyhow::Result;
use std::sync::{Arc, Mutex, RwLock};

/// Generic implementation of VideoFrame
pub struct GenericVideoFrame {
    native: NativeVideoFrame,
}

impl GenericVideoFrame {
    /// Create a new GenericVideoFrame from a NativeVideoFrame
    pub fn from_native(native: NativeVideoFrame) -> Self {
        Self { native }
    }

    /// Create a new GenericVideoFrame from raw data
    pub fn from_raw(data: Vec<u8>, width: u32, height: u32, timestamp_us: i64) -> Self {
        Self {
            native: NativeVideoFrame::new(
                data,
                width,
                height,
                timestamp_us,
                VideoRotation::Rotation0,
            ),
        }
    }
}

impl VideoFrameImpl for GenericVideoFrame {
    fn to_rgba_bytes(&self) -> Result<Vec<u8>> {
        self.native.to_rgba_bytes()
    }

    fn width(&self) -> u32 {
        self.native.width()
    }

    fn height(&self) -> u32 {
        self.native.height()
    }

    fn timestamp_us(&self) -> i64 {
        self.native.timestamp_us()
    }
}

/// Generic implementation of VideoSource
pub struct GenericVideoSource {
    info: VideoSourceInfo,
    current_frame: Arc<RwLock<Option<VideoFrame>>>,
    is_active: bool,
    frame_timer: Option<tokio::task::JoinHandle<()>>,
}

impl GenericVideoSource {
    /// Create a new GenericVideoSource from a camera
    pub fn from_camera(options: VideoSourceOptions) -> Result<Self> {
        let width = options.width.unwrap_or(640);
        let height = options.height.unwrap_or(480);
        let fps = options.fps.unwrap_or(30) as f64;

        Ok(Self {
            info: VideoSourceInfo {
                name: "Generic Camera".to_string(),
                width,
                height,
                fps,
                is_active: false,
            },
            current_frame: Arc::new(RwLock::new(None)),
            is_active: false,
            frame_timer: None,
        })
    }

    /// Create a new GenericVideoSource from screen capture
    pub fn from_screen(options: VideoSourceOptions) -> Result<Self> {
        let width = options.width.unwrap_or(1920);
        let height = options.height.unwrap_or(1080);
        let fps = options.fps.unwrap_or(30) as f64;

        Ok(Self {
            info: VideoSourceInfo {
                name: "Generic Screen".to_string(),
                width,
                height,
                fps,
                is_active: false,
            },
            current_frame: Arc::new(RwLock::new(None)),
            is_active: false,
            frame_timer: None,
        })
    }

    /// Create a new GenericVideoSource from a file
    pub fn from_file(path: &str, options: VideoSourceOptions) -> Result<Self> {
        let width = options.width.unwrap_or(1280);
        let height = options.height.unwrap_or(720);
        let fps = options.fps.unwrap_or(30) as f64;

        Ok(Self {
            info: VideoSourceInfo {
                name: format!("Generic File: {}", path),
                width,
                height,
                fps,
                is_active: false,
            },
            current_frame: Arc::new(RwLock::new(None)),
            is_active: false,
            frame_timer: None,
        })
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

        let generic_frame = GenericVideoFrame::from_raw(
            data,
            width,
            height,
            frame_number as i64 * 1_000_000 / self.info.fps as i64,
        );

        VideoFrame::new(generic_frame)
    }
}

impl VideoSourceImpl for GenericVideoSource {
    fn start(&mut self) -> Result<()> {
        if self.is_active {
            return Ok(());
        }

        self.is_active = true;
        self.info.is_active = true;

        // Start generating frames
        let current_frame = self.current_frame.clone();
        let info = self.info.clone();

        let frame_timer = tokio::spawn(async move {
            let mut frame_number = 0;

            let frame_duration = std::time::Duration::from_millis((1000.0 / info.fps) as u64);

            loop {
                // Generate a test pattern frame
                let frame = {
                    let generic_source = GenericVideoSource {
                        info: info.clone(),
                        current_frame: current_frame.clone(),
                        is_active: true,
                        frame_timer: None,
                    };

                    generic_source.generate_test_pattern(frame_number)
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

    fn stop(&mut self) -> Result<()> {
        if !self.is_active {
            return Ok(());
        }

        self.is_active = false;
        self.info.is_active = false;

        // Stop frame generation
        if let Some(timer) = self.frame_timer.take() {
            timer.abort();
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
