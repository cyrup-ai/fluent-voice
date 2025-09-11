use crate::VideoFrame;
use crate::VideoSource;
use anyhow::Result;
use futures::{Stream, StreamExt};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

/// Represents a video track that can be played, recorded, or transmitted
#[derive(Clone)]
pub struct VideoTrack {
    source: VideoSource,
    current_frame: Arc<RwLock<Option<VideoFrame>>>,
    frame_processor: Arc<Mutex<Option<Box<dyn FrameProcessor + Send + Sync>>>>,
}

/// Trait for processing video frames
pub trait FrameProcessor {
    /// Process a video frame
    fn process(&mut self, frame: &VideoFrame) -> Result<VideoFrame>;
}

impl std::fmt::Debug for VideoTrack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoTrack")
            .field("source", &self.source)
            .field("current_frame", &"<frame data>")
            .field("frame_processor", &"<processor>")
            .finish()
    }
}

impl VideoTrack {
    /// Create a new video track from a video source
    pub fn new(source: VideoSource) -> Self {
        Self {
            source,
            current_frame: Arc::new(RwLock::new(None)),
            frame_processor: Arc::new(Mutex::new(None)),
        }
    }

    /// Start playing the video track
    pub fn play(&self) -> Result<()> {
        self.source.start()?;

        // Start frame update loop
        let source = self.source.clone();
        let current_frame = self.current_frame.clone();
        let frame_processor = self.frame_processor.clone();

        tokio::spawn(async move {
            loop {
                if let Some(frame) = source.get_current_frame() {
                    // Apply frame processor if one is set
                    let processed_frame = {
                        match frame_processor.lock() {
                            Ok(mut processor_guard) => {
                                if let Some(processor) = processor_guard.as_mut() {
                                    processor.process(&frame).unwrap_or(frame)
                                } else {
                                    frame
                                }
                            }
                            Err(_) => {
                                // Failed to acquire lock, use original frame
                                frame
                            }
                        }
                    };

                    // Update current frame
                    if let Ok(mut write_guard) = current_frame.write() {
                        *write_guard = Some(processed_frame);
                    }
                }

                tokio::time::sleep(Duration::from_millis(16)).await; // ~60fps
            }
        });

        Ok(())
    }

    /// Stop playing the video track
    pub fn stop(&self) -> Result<()> {
        self.source.stop()
    }

    /// Get the current frame from the video track
    pub fn get_current_frame(&self) -> Option<VideoFrame> {
        if let Ok(read_guard) = self.current_frame.read() {
            read_guard.clone()
        } else {
            None
        }
    }

    /// Set a frame processor for the video track
    pub fn set_frame_processor(&self, processor: Box<dyn FrameProcessor + Send + Sync>) {
        if let Ok(mut write_guard) = self.frame_processor.lock() {
            *write_guard = Some(processor);
        }
    }

    /// Remove the frame processor from the video track
    pub fn remove_frame_processor(&self) {
        if let Ok(mut write_guard) = self.frame_processor.lock() {
            *write_guard = None;
        }
    }

    /// Get a stream of video frames from the video track
    pub fn get_frame_stream(&self) -> impl Stream<Item = VideoFrame> + Send + 'static {
        let current_frame = self.current_frame.clone();

        futures::stream::unfold(current_frame, |current_frame| async move {
            let frame = if let Ok(read_guard) = current_frame.read() {
                read_guard.clone()
            } else {
                None
            };

            tokio::time::sleep(Duration::from_millis(16)).await; // ~60fps

            if let Some(frame) = frame {
                Some((frame, current_frame))
            } else {
                Some((VideoFrame::default(), current_frame))
            }
        })
        .filter(|frame| futures::future::ready(!frame.is_empty()))
    }
}
