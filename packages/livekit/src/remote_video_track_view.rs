// Temporarily disabled - external dependency on ratagpu
// This module will be restored once ratagpu is properly integrated

use super::livekit_client::RemoteVideoTrack;
use futures::channel::mpsc;
use std::{
    sync::{Arc, RwLock},
    time::Duration,
};
use tokio::task::JoinHandle;

#[allow(dead_code)]
pub struct RemoteVideoTrackView {
    track: RemoteVideoTrack,
    latest_frame: Arc<RwLock<Option<crate::RemoteVideoFrame>>>,
    window: Option<Arc<winit::window::Window>>,
    _frame_processor: JoinHandle<()>,
    event_sender: mpsc::UnboundedSender<RemoteVideoTrackViewEvent>,
    event_receiver: mpsc::UnboundedReceiver<RemoteVideoTrackViewEvent>,
    scale_factor: f64,
}

pub enum RemoteVideoTrackViewEvent {
    // Add event types as needed
}

impl RemoteVideoTrackView {
    pub fn new(track: RemoteVideoTrack) -> Self {
        let (event_sender, event_receiver) = mpsc::unbounded();
        // CVPixelBuffer is not Send/Sync by design (Core Video framework constraint)
        // This Arc will be replaced with proper thread-safe video frame handling
        // when ratagpu integration is restored
        #[allow(clippy::arc_with_non_send_sync)]
        let latest_frame = Arc::new(RwLock::new(None));

        // Placeholder frame processor
        let _frame_processor = tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(1)).await;
        });

        Self {
            track,
            latest_frame,
            window: None,
            _frame_processor,
            event_sender,
            event_receiver,
            scale_factor: 1.0,
        }
    }

    pub fn initialize_renderer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Renderer initialization disabled - external dependency
        Ok(())
    }

    pub fn render_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Frame rendering disabled - external dependency
        Ok(())
    }

    pub fn handle_resize(
        &mut self,
        _width: u32,
        _height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Resize handling disabled - external dependency
        Ok(())
    }
}
