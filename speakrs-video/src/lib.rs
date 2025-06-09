pub mod cli_args;
mod native_video;
mod track;
mod video_frame;
mod video_source;

use track::*;
use video_frame::*;
use video_source::*;

#[cfg(target_os = "macos")]
mod macos;

#[cfg(not(target_os = "macos"))]
mod generic;

pub use track::VideoTrack;
pub use video_frame::VideoFrame;
pub use video_source::VideoSource;

use raw_window_handle::RawWindowHandle;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct VideoTrackView {
    track: VideoTrack,
    renderer: Option<Arc<Mutex<sugarloaf::SugarloafRenderer>>>,
    graphic_id: Option<sugarloaf::graphics::GraphicId>,
}

impl VideoTrackView {
    pub fn new(track: VideoTrack) -> Self {
        Self {
            track,
            renderer: None,
            graphic_id: None,
        }
    }

    pub fn initialize_renderer(&mut self, window_handle: RawWindowHandle) -> anyhow::Result<()> {
        let window = sugarloaf::SugarloafWindow::Raw(window_handle);
        let mut sugarloaf =
            sugarloaf::Sugarloaf::new(window, None, wgpu::TextureFormat::Bgra8UnormSrgb, None)?;

        // Create initial empty graphic
        let graphic_id = sugarloaf.create_graphic(500, 500)?;

        // Store sugarloaf and renderer
        let renderer = sugarloaf.create_renderer();

        self.renderer = Some(Arc::new(Mutex::new(renderer)));
        self.graphic_id = Some(graphic_id);

        Ok(())
    }

    pub fn update(&mut self) -> anyhow::Result<()> {
        if let Some(frame) = self.track.get_current_frame() {
            if let Some(graphic_id) = self.graphic_id {
                if let Some(renderer) = &mut self.renderer {
                    let mut renderer = renderer.lock().unwrap();

                    // Update graphic with new frame data
                    let graphic_data = frame.to_rgba_bytes()?;
                    let width = frame.width();
                    let height = frame.height();

                    renderer.update_graphic(
                        graphic_id,
                        &sugarloaf::graphics::GraphicData::Rgba8(graphic_data),
                        width,
                        height,
                    )?;

                    // Render current frame
                    renderer.render()?;
                }
            }
        }

        Ok(())
    }

    pub fn handle_resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        if let Some(renderer) = &mut self.renderer {
            let mut renderer = renderer.lock().unwrap();
            renderer.resize(width, height)?;
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
