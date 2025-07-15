use super::livekit_client::RemoteVideoTrack;
use crate::playback::VideoFrameExtensions;
use crate::util::ResultExt;
use futures::{StreamExt as _, channel::mpsc};
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::{
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};
use tokio::task::JoinHandle;
// use wgpu::TextureFormat; // Using ratagpu instead which has wgpu integrated

pub struct RemoteVideoTrackView {
    track: RemoteVideoTrack,
    latest_frame: Arc<RwLock<Option<crate::RemoteVideoFrame>>>,
    renderer: Option<Arc<Mutex<ratagpu::ZeroAllocRenderer<80, 24>>>>,
    window: Option<Arc<winit::window::Window>>,
    _frame_processor: JoinHandle<()>,
    event_sender: mpsc::UnboundedSender<RemoteVideoTrackViewEvent>,
    event_receiver: mpsc::UnboundedReceiver<RemoteVideoTrackViewEvent>,
    scale_factor: f64,
}

#[derive(Debug, Clone)]
pub enum RemoteVideoTrackViewEvent {
    Close,
    FrameUpdated,
}

impl RemoteVideoTrackView {
    pub fn new(track: RemoteVideoTrack, window: Arc<winit::window::Window>) -> Self {
        let frames = super::play_remote_video_track(&track);
        let latest_frame = Arc::new(RwLock::new(None));
        let latest_frame_clone = latest_frame.clone();

        let (tx, rx) = mpsc::unbounded();
        let tx_clone = tx.clone();

        let frame_processor = tokio::spawn(async move {
            futures::pin_mut!(frames);
            while let Some(frame) = frames.next().await {
                // Update the latest frame
                if let Ok(mut write_guard) = latest_frame_clone.write() {
                    *write_guard = Some(frame);

                    // Notify about frame update
                    let _ = tx_clone.unbounded_send(RemoteVideoTrackViewEvent::FrameUpdated);
                }

                // Small sleep to prevent CPU overuse
                tokio::time::sleep(Duration::from_millis(16)).await;
            }

            // Notify when the stream ends
            let _ = tx_clone.unbounded_send(RemoteVideoTrackViewEvent::Close);
        });

        Self {
            track,
            latest_frame,
            renderer: None,
            window: Some(window),
            _frame_processor: frame_processor,
            event_sender: tx,
            event_receiver: rx,
            scale_factor: 1.0,
        }
    }

    pub fn initialize_renderer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(window) = &self.window {
            // Create Ratagpu renderer for terminal-style video display
            let renderer = pollster::block_on(async {
                ratagpu::RendererBuilder::<80, 24>::new()
                    .build(window.as_ref())
                    .await
            })?;

            self.renderer = Some(Arc::new(Mutex::new(renderer)));
        }
        Ok(())
    }

    pub fn update(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        while let Ok(Some(event)) = self.event_receiver.try_next() {
            match event {
                RemoteVideoTrackViewEvent::FrameUpdated => {
                    self.update_frame()?;
                }
                RemoteVideoTrackViewEvent::Close => {
                    // Handle close event
                }
            }
        }

        // Render the frame if we have a renderer
        if let Some(renderer) = &self.renderer {
            if let Ok(mut renderer) = renderer.lock() {
                renderer.render()?;
            }
        }

        Ok(())
    }

    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    pub fn set_scale_factor(&mut self, scale_factor: f64) {
        self.scale_factor = scale_factor;
    }

    pub fn track(&self) -> &RemoteVideoTrack {
        &self.track
    }

    pub fn current_frame_size(&self) -> Option<(u32, u32)> {
        if let Ok(guard) = self.latest_frame.read() {
            if let Some(frame) = &*guard {
                return Some((frame.width(), frame.height()));
            }
        }
        None
    }

    pub fn get_ascii_buffer(&self) -> Vec<String> {
        // This method is kept for compatibility but returns empty since Ratagpu handles rendering
        Vec::new()
    }

    pub fn update_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(guard) = self.latest_frame.read() {
            if let Some(frame) = &*guard {
                if let Some(renderer) = &self.renderer {
                    if let Ok(mut renderer) = renderer.lock() {
                        // Convert video frame to ASCII art representation
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

                                    renderer.set_cell(
                                        x,
                                        y,
                                        ratagpu::Cell {
                                            character: ch,
                                            foreground: 15, // White
                                            background: 0,  // Black
                                            style_flags: 0,
                                        },
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn handle_resize(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(renderer) = &self.renderer {
            if let Ok(mut renderer) = renderer.lock() {
                renderer.handle_resize(width, height)?;
            }
        }

        Ok(())
    }

    pub fn clone(&self) -> Self {
        // Create a new view with the same track
        let window = self.window.clone().unwrap_or_else(|| {
            // This is a fallback - in practice, window should always be available
            panic!("Cannot clone RemoteVideoTrackView without window")
        });
        let mut new_view = Self::new(self.track.clone(), window);
        new_view.scale_factor = self.scale_factor;
        new_view
    }

    pub fn send_event(
        &self,
        event: RemoteVideoTrackViewEvent,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.event_sender
            .unbounded_send(event)
            .map_err(|e| e.into())
    }

    pub fn receive_events(
        &mut self,
    ) -> Result<Vec<RemoteVideoTrackViewEvent>, Box<dyn std::error::Error>> {
        let mut events = Vec::new();
        while let Ok(Some(event)) = self.event_receiver.try_next() {
            events.push(event);
        }
        Ok(events)
    }
}

impl Drop for RemoteVideoTrackView {
    fn drop(&mut self) {
        // Clean up resources - renderer cleanup is handled automatically
    }
}
