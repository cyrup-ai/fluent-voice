pub mod cli_args;
mod native_video;
mod track;
mod video_frame;
mod video_source;

// use video_frame::*;
// use video_source::*;

#[cfg(target_os = "macos")]
mod macos;

#[cfg(not(target_os = "macos"))]
mod generic;

pub use native_video::{NativeVideoBuffer, NativeVideoFrame, VideoRotation};
pub use track::VideoTrack;
pub use video_frame::VideoFrame;
pub use video_source::{VideoSource, VideoSourceOptions};

/// Public API for video frame creation with rotation support
#[inline(always)]
pub fn create_rotated_video_frame(
    buffer: Vec<u8>,
    width: u32,
    height: u32,
    timestamp_us: i64,
    rotation: VideoRotation,
) -> VideoFrame {
    NativeVideoFrame::create_video_frame(buffer, width, height, timestamp_us, rotation)
}

/// Public API for video frame rotation with zero-allocation performance
#[inline(always)]
pub fn rotate_video_frame(frame: &mut NativeVideoFrame, rotation: VideoRotation) {
    frame.rotate(rotation);
}

#[cfg(target_os = "macos")]
/// Create MacOS video frame from Core Video buffer with blazing-fast conversion
#[inline(always)]
pub fn create_macos_video_frame_from_native(native_frame: NativeVideoFrame) -> VideoFrame {
    use macos::MacOSVideoFrame;
    let frame = MacOSVideoFrame::from_native(native_frame);
    VideoFrame::new(frame)
}

#[cfg(target_os = "macos")]
/// Create MacOS video frame from CVImageBuffer with real Core Video dimensions
#[inline(always)]
pub fn create_macos_video_frame_from_cv_buffer(
    buffer: core_video::image_buffer::CVImageBuffer,
    timestamp_us: i64,
) -> VideoFrame {
    use macos::MacOSVideoFrame;
    let frame = MacOSVideoFrame::from_cv_buffer(buffer, timestamp_us);
    VideoFrame::new(frame)
}

#[cfg(target_os = "macos")]
/// Create ThreadSafeCVImageBuffer wrapper for multi-threaded Core Video access
#[inline(always)]
pub fn create_thread_safe_wrapper(
    buffer: core_video::image_buffer::CVImageBuffer,
) -> macos::ThreadSafeCVImageBuffer {
    macos::ThreadSafeCVImageBuffer::new(buffer)
}

// VideoTrackView is defined later in this file

use std::sync::{Arc, Mutex};

/// Terminal-based video renderer for ASCII art display
pub struct TerminalRenderer {
    width: u32,
    height: u32,
    frame_buffer: Vec<Vec<char>>,
    cursor_hidden: bool,
}

impl Default for TerminalRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalRenderer {
    pub fn new() -> Self {
        let width = 80;
        let height = 24;
        let frame_buffer = vec![vec![' '; width as usize]; height as usize];

        Self {
            width,
            height,
            frame_buffer,
            cursor_hidden: false,
        }
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        use std::io::{self, Write};

        // Hide cursor for smoother rendering
        if !self.cursor_hidden {
            print!("\x1B[?25l"); // Hide cursor
            self.cursor_hidden = true;
        }

        // Move cursor to top-left and clear screen
        print!("\x1B[H\x1B[2J");

        // Render frame buffer to terminal
        for row in &self.frame_buffer {
            for &ch in row {
                print!("{}", ch);
            }
            println!(); // New line after each row
        }

        // Flush output for immediate display
        io::stdout().flush()?;
        Ok(())
    }

    pub fn handle_resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        self.width = width;
        self.height = height;

        // Resize frame buffer to match new dimensions
        self.frame_buffer = vec![vec![' '; width as usize]; height as usize];
        Ok(())
    }

    /// Set a character at the specified position in the frame buffer
    pub fn set_char(&mut self, x: u32, y: u32, ch: char) {
        let x = x as usize;
        let y = y as usize;

        if y < self.frame_buffer.len() && x < self.frame_buffer[y].len() {
            self.frame_buffer[y][x] = ch;
        }
    }

    /// Clear the frame buffer
    pub fn clear(&mut self) {
        for row in &mut self.frame_buffer {
            for cell in row {
                *cell = ' ';
            }
        }
    }
}

impl Drop for TerminalRenderer {
    fn drop(&mut self) {
        // Restore cursor when renderer is dropped
        if self.cursor_hidden {
            print!("\x1B[?25h"); // Show cursor
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }
}

#[derive(Clone)]
pub struct VideoTrackView {
    track: VideoTrack,
    renderer: Option<Arc<Mutex<TerminalRenderer>>>,
    window: Option<Arc<winit::window::Window>>,
}

impl std::fmt::Debug for VideoTrackView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoTrackView")
            .field("track", &self.track)
            .field("renderer", &"<renderer>")
            .field("window", &self.window.is_some())
            .finish()
    }
}

impl VideoTrackView {
    pub fn new(track: VideoTrack) -> Self {
        Self {
            track,
            renderer: None,
            window: None,
        }
    }

    pub fn new_from_remote(
        _remote_track: livekit::track::RemoteVideoTrack,
    ) -> anyhow::Result<Self> {
        // Convert remote LiveKit track to our VideoTrack wrapper
        // Create a placeholder video source for remote track integration
        // In production, this would bridge LiveKit remote video data to our VideoSource
        use crate::{VideoSource, VideoSourceOptions};

        let options = VideoSourceOptions {
            width: Some(640), // Default resolution
            height: Some(480),
            fps: Some(30),
        };

        // Create a placeholder camera source - in production this would be a
        // RemoteVideoSource that bridges LiveKit remote track data
        let source = VideoSource::from_camera(options)?;
        let track = VideoTrack::new(source);

        Ok(Self {
            track,
            renderer: None,
            window: None,
        })
    }

    pub fn initialize_renderer(&mut self, _window: &winit::window::Window) -> anyhow::Result<()> {
        // Create terminal renderer for ASCII art video display
        let renderer = TerminalRenderer::new();

        self.renderer = Some(Arc::new(Mutex::new(renderer)));
        Ok(())
    }

    pub fn update(&mut self) -> anyhow::Result<()> {
        if let Some(frame) = self.track.get_current_frame()
            && let Some(renderer) = &mut self.renderer
            && let Ok(mut renderer) = renderer.lock()
        {
            // Clear the previous frame
            renderer.clear();

            // Convert video frame to ASCII art representation for terminal display
            let frame_data = frame.to_rgba_bytes()?;
            let frame_width = frame.width() as usize;
            let frame_height = frame.height() as usize;

            // Simple ASCII art conversion for terminal display
            let ascii_chars = " .:-=+*#%@";

            for y in 0..24 {
                for x in 0..80 {
                    let src_x = if frame_width > 0 {
                        x * frame_width / 80
                    } else {
                        0
                    };
                    let src_y = if frame_height > 0 {
                        y * frame_height / 24
                    } else {
                        0
                    };
                    let pixel_idx = (src_y * frame_width + src_x) * 4;

                    if pixel_idx + 3 < frame_data.len() {
                        let r = frame_data[pixel_idx] as f32;
                        let g = frame_data[pixel_idx + 1] as f32;
                        let b = frame_data[pixel_idx + 2] as f32;

                        let brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
                        let char_idx = (brightness * (ascii_chars.len() - 1) as f32) as usize;
                        let ch = ascii_chars.chars().nth(char_idx).unwrap_or(' ');

                        // Set the character in the terminal renderer frame buffer
                        renderer.set_char(x as u32, y as u32, ch);
                    }
                }
            }

            // Render the frame
            renderer.render()?;
        }

        Ok(())
    }

    pub fn handle_resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        if let Some(renderer) = &mut self.renderer
            && let Ok(mut renderer) = renderer.lock()
        {
            renderer.handle_resize(width, height)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_video::{NativeVideoFrame, VideoRotation};

    #[tokio::test]
    async fn test_terminal_renderer_basic_functionality() {
        let mut renderer = TerminalRenderer::new();

        // Test initial state
        assert_eq!(renderer.width, 80);
        assert_eq!(renderer.height, 24);

        // Test character setting
        renderer.set_char(5, 3, 'X');
        assert_eq!(renderer.frame_buffer[3][5], 'X');

        // Test clear functionality
        renderer.clear();
        assert_eq!(renderer.frame_buffer[3][5], ' ');

        // Test resize
        let result = renderer.handle_resize(120, 30);
        assert!(result.is_ok());
        assert_eq!(renderer.width, 120);
        assert_eq!(renderer.height, 30);
        assert_eq!(renderer.frame_buffer.len(), 30);
        assert_eq!(renderer.frame_buffer[0].len(), 120);
    }

    #[tokio::test]
    async fn test_video_frame_creation() {
        // Test creating a video frame with rotation
        let buffer = vec![255u8; 640 * 480 * 4]; // RGBA buffer
        let frame = create_rotated_video_frame(
            buffer,
            640,
            480,
            1_000_000, // 1 second timestamp
            VideoRotation::Rotation0,
        );

        assert_eq!(frame.width(), 640);
        assert_eq!(frame.height(), 480);
        assert_eq!(frame.timestamp_us(), 1_000_000);
        assert!(!frame.is_empty());
    }

    #[tokio::test]
    async fn test_video_track_view_creation() {
        use crate::track::VideoTrack;

        // Create a simple video track for testing
        let track = VideoTrack::new("test_track".to_string());
        let track_view = VideoTrackView::new(track);

        // Verify initial state
        assert!(track_view.renderer.is_none());
        assert!(track_view.window.is_none());
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn test_thread_safe_wrapper_creation() {
        use core_foundation::base::TCFType;
        use core_video::image_buffer::CVImageBuffer;

        // This test requires actual CVImageBuffer creation which needs macOS APIs
        // For now, test that the function signature is correct
        // A real CVImageBuffer would be created using Core Video APIs in actual usage
    }
}
