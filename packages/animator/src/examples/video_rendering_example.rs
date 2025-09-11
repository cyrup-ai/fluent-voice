use std::sync::Arc;
use egui::Vec2;
use egui_wgpu::RenderState;
use livekit::webrtc::prelude::*;
use tokio::runtime::Runtime;
use crate::VideoRenderer;

pub fn run_video_rendering_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a minimal runtime for the example
    let runtime = Runtime::new()?;
    
    // Create a mock video track for demonstration
    let video_track = create_mock_video_track();
    
    // Create egui native options
    let native_options = eframe::NativeOptions {
        initial_window_size: Some(Vec2::new(800.0, 600.0)),
        ..Default::default()
    };
    
    // Run the egui application
    eframe::run_native(
        "Video Rendering Example",
        native_options,
        Box::new(|cc| {
            // Get the wgpu render state from the eframe creation context
            let wgpu_render_state = cc.wgpu_render_state.as_ref().expect("WGPU must be enabled");
            
            // Create the video renderer
            let video_renderer = VideoRenderer::new(
                runtime.handle(),
                wgpu_render_state.clone(),
                video_track,
            );
            
            Box::new(VideoRendererApp { video_renderer })
        }),
    )?;
    
    Ok(())
}

struct VideoRendererApp {
    video_renderer: VideoRenderer,
}

impl eframe::App for VideoRendererApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Video Renderer");
            ui.add_space(20.0);
            
            // Display video frame
            if let Some(texture_id) = self.video_renderer.texture_id() {
                let (width, height) = self.video_renderer.resolution();
                
                // Calculate the display size while preserving aspect ratio
                let available_width = ui.available_width().min(640.0);
                let available_height = ui.available_height().min(480.0);
                
                let aspect_ratio = width as f32 / height as f32;
                let display_size = if available_width / aspect_ratio <= available_height {
                    // Width constrained
                    Vec2::new(available_width, available_width / aspect_ratio)
                } else {
                    // Height constrained
                    Vec2::new(available_height * aspect_ratio, available_height)
                };
                
                // Display the video frame
                ui.image(texture_id, display_size);
            } else {
                // Display a placeholder if no video frame is available
                let placeholder_size = Vec2::new(640.0, 480.0);
                let (rect, _response) = ui.allocate_exact_size(placeholder_size, egui::Sense::hover());
                ui.painter().rect_filled(
                    rect,
                    0.0,
                    egui::Color32::from_gray(40),
                );
                
                // Add "No Video" text
                let text_layout = egui::TextLayout::from_galley(
                    ui.fonts(|f| f.layout_single_line("No Video Signal", f32::INFINITY, 24.0)),
                    egui::Color32::from_gray(200),
                    egui::Align2::CENTER_CENTER,
                );
                ui.painter().add(egui::Shape::text(
                    text_layout, 
                    rect.center(),
                ));
            }
            
            ui.add_space(20.0);
            ui.label("The video renderer displays frames from a WebRTC video track.");
            ui.label("In a real application, this would show a live video stream from a LiveKit room.");
            
            // Add some interactive controls as an example
            ui.add_space(10.0);
            ui.separator();
            ui.add_space(10.0);
            
            ui.horizontal(|ui| {
                ui.label("Brightness:");
                ui.add(egui::Slider::new(&mut 1.0, 0.5..=1.5).step_by(0.1));
            });
            
            ui.horizontal(|ui| {
                ui.label("Contrast:");
                ui.add(egui::Slider::new(&mut 1.0, 0.5..=1.5).step_by(0.1));
            });
            
            ui.horizontal(|ui| {
                ui.label("Zoom:");
                ui.add(egui::Slider::new(&mut 1.0, 1.0..=3.0).step_by(0.1));
            });
            
            // Note: These controls don't actually modify the video in this example
            // In a real implementation, you would apply these transformations to the video
        });
        
        // Request continuous redraw to update the video
        ctx.request_repaint();
    }
}

// Helper function to create a mock video track for this example
fn create_mock_video_track() -> RtcVideoTrack {
    // In a real application, you would get this from a LiveKit room
    // For this example, we create a mock that simulates video data
    
    // This is a simplified mock implementation - in a real application,
    // you would use actual LiveKit tracks
    let track_info = TrackInfo::new("mock_video_track".to_string(), TrackType::Video);
    RtcVideoTrack::new(Arc::new(track_info))
}