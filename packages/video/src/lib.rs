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

pub use track::VideoTrack;
pub use video_frame::VideoFrame;
pub use video_source::{VideoSource, VideoSourceOptions};

// VideoTrackView is defined later in this file

use std::sync::{Arc, Mutex};

// Placeholder renderer until ratagpu is available
pub struct PlaceholderRenderer {
    width: u32,
    height: u32,
}

impl PlaceholderRenderer {
    pub fn new() -> Self {
        Self {
            width: 80,
            height: 24,
        }
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        // Placeholder implementation
        Ok(())
    }

    pub fn handle_resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        self.width = width;
        self.height = height;
        Ok(())
    }
}

#[derive(Clone)]
pub struct VideoTrackView {
    track: VideoTrack,
    renderer: Option<Arc<Mutex<PlaceholderRenderer>>>,
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

    pub fn initialize_renderer(&mut self, _window: &winit::window::Window) -> anyhow::Result<()> {
        // Create placeholder renderer until ratagpu is available
        let renderer = PlaceholderRenderer::new();

        self.renderer = Some(Arc::new(Mutex::new(renderer)));
        Ok(())
    }

    pub fn update(&mut self) -> anyhow::Result<()> {
        if let Some(frame) = self.track.get_current_frame() {
            if let Some(renderer) = &mut self.renderer {
                if let Ok(mut renderer) = renderer.lock() {
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
                                let char_idx =
                                    (brightness * (ascii_chars.len() - 1) as f32) as usize;
                                let ch = ascii_chars.chars().nth(char_idx).unwrap_or(' ');

                                // TODO: Implement actual cell rendering when ratagpu is available
                                // For now, just store the character (placeholder)
                                let _cell_data = (x, y, ch);
                            }
                        }
                    }

                    // Render the frame
                    renderer.render()?;
                }
            }
        }

        Ok(())
    }

    pub fn handle_resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        if let Some(renderer) = &mut self.renderer {
            if let Ok(mut renderer) = renderer.lock() {
                renderer.handle_resize(width, height)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_track_view() {
        // This test will be implemented when we have proper mocks
    }
}
