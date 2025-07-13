use super::livekit_client::RemoteVideoTrack;
use crate::playback::VideoFrameExtensions;
use crate::util::ResultExt;
use futures::{StreamExt as _, channel::mpsc};
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::{
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};
use sugarloaf::{
    Sugarloaf, SugarloafRenderer, SugarloafWindow,
    graphics::{Graphic, GraphicId},
};
use tokio::task::JoinHandle;
use wgpu::TextureFormat;

pub struct RemoteVideoTrackView {
    track: RemoteVideoTrack,
    latest_frame: Arc<RwLock<Option<crate::RemoteVideoFrame>>>,
    sugarloaf: Arc<Mutex<Option<Sugarloaf>>>,
    renderer: Option<SugarloafRenderer>,
    graphic_id: Option<GraphicId>,
    window_handle: RawWindowHandle,
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
    pub fn new<W>(track: RemoteVideoTrack, window: &W) -> Self
    where
        W: HasRawWindowHandle + ?Sized,
    {
        let frames = super::play_remote_video_track(&track);
        let latest_frame = Arc::new(RwLock::new(None));
        let latest_frame_clone = latest_frame.clone();

        let (tx, rx) = mpsc::unbounded();
        let tx_clone = tx.clone();

        let window_handle = window.raw_window_handle();

        // Setup Sugarloaf renderer
        let sugarloaf = Arc::new(Mutex::new(None));
        let sugarloaf_clone = sugarloaf.clone();

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
            sugarloaf,
            renderer: None,
            graphic_id: None,
            window_handle,
            _frame_processor: frame_processor,
            event_sender: tx,
            event_receiver: rx,
            scale_factor: 1.0,
        }
    }

    pub fn initialize_renderer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Initialize Sugarloaf for rendering
        let window = SugarloafWindow::Raw(self.window_handle);
        let mut sugarloaf = Sugarloaf::new(window, None, TextureFormat::Bgra8UnormSrgb, None)?;

        // Create initial empty graphic
        let graphic_id = sugarloaf.create_graphic(500, 500)?;

        // Store sugarloaf and renderer
        let renderer = sugarloaf.create_renderer();

        if let Ok(mut guard) = self.sugarloaf.lock() {
            *guard = Some(sugarloaf);
        }

        self.renderer = Some(renderer);
        self.graphic_id = Some(graphic_id);

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

        // Render current frame
        if let Some(renderer) = &mut self.renderer {
            renderer.render()?;
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

    pub fn renderer(&self) -> Option<&SugarloafRenderer> {
        self.renderer.as_ref()
    }

    pub fn renderer_mut(&mut self) -> Option<&mut SugarloafRenderer> {
        self.renderer.as_mut()
    }

    pub fn graphic_id(&self) -> Option<GraphicId> {
        self.graphic_id
    }

    pub fn sugarloaf(&self) -> Arc<Mutex<Option<Sugarloaf>>> {
        self.sugarloaf.clone()
    }

    pub fn update_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(guard) = self.latest_frame.read() {
            if let Some(frame) = &*guard {
                if let Some(graphic_id) = self.graphic_id {
                    if let Ok(mut guard) = self.sugarloaf.lock() {
                        if let Some(sugarloaf) = &mut *guard {
                            // Update graphic with new frame data using VideoFrameExtensions trait
                            let graphic_data = frame.to_rgba_bytes()?;
                            let width = frame.width();
                            let height = frame.height();

                            sugarloaf.update_graphic(
                                graphic_id,
                                &sugarloaf::graphics::GraphicData::Rgba8(graphic_data),
                                width,
                                height,
                            )?;
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
        if let Ok(mut guard) = self.sugarloaf.lock() {
            if let Some(sugarloaf) = &mut *guard {
                sugarloaf.resize(width, height)?;
            }
        }

        Ok(())
    }

    pub fn clone(&self) -> Self {
        // Create a new view with the same track
        let mut new_view = Self::new(self.track.clone(), &mut ());

        // Clone graphic_id if possible
        new_view.graphic_id = self.graphic_id;
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
        // Clean up resources
        if let Some(graphic_id) = self.graphic_id {
            if let Ok(mut guard) = self.sugarloaf.lock() {
                if let Some(sugarloaf) = &mut *guard {
                    let _ = sugarloaf.remove_graphic(graphic_id);
                }
            }
        }
    }
}
