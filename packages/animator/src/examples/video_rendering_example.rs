use crate::VideoRenderer;
use egui::Vec2;
use egui_wgpu::RenderState;
use fluent_voice_livekit::{RemoteTrack, RemoteVideoTrack, Room, RoomEvent};
use futures::StreamExt;
use livekit::webrtc::prelude::*;
use livekit_api::access_token;
use std::{env, sync::Arc, time::Duration};
use tokio::runtime::Runtime;

async fn create_livekit_video_track(
    room_url: &str,
    access_token: &str,
) -> Result<livekit::webrtc::prelude::RtcVideoTrack, Box<dyn std::error::Error>> {
    // Connect to LiveKit room with timeout
    let (room, mut events) = tokio::time::timeout(
        Duration::from_secs(10),
        Room::connect(room_url.to_string(), access_token.to_string())
    ).await??;

    // Wait for first video track with timeout
    let timeout_duration = Duration::from_secs(30);
    let start_time = std::time::Instant::now();
    
    while let Some(event) = events.next().await {
        if start_time.elapsed() > timeout_duration {
            return Err("Timeout waiting for video track".into());
        }
        match event {
            RoomEvent::TrackSubscribed {
                track: RemoteTrack::Video(video_track),
                publication,
                participant,
            } => {
                tracing::info!(
                    participant = %participant.identity().0,
                    track_sid = %video_track.sid(),
                    "Connected to remote video track"
                );

                // Enable the track
                publication.set_enabled(true);

                // Return the underlying RTC track
                return Ok(video_track.0.clone());
            }
            RoomEvent::Connected {
                participants_with_tracks,
            } => {
                // Check for existing video tracks
                for (participant, publications) in participants_with_tracks {
                    for publication in publications {
                        if !publication.is_audio() {
                            if let Some(RemoteTrack::Video(track)) = publication.track() {
                                tracing::info!(
                                    participant = %participant.identity().0,
                                    "Found existing video track"
                                );
                                publication.set_enabled(true);
                                return Ok(track.0.clone());
                            }
                        }
                    }
                }
            }
            _ => continue,
        }
    }

    Err("No video track found in room".into())
}

pub fn run_video_rendering_example() -> Result<(), Box<dyn std::error::Error>> {
    // Get connection details from environment variables
    let room_url = env::var("LIVEKIT_URL")
        .or_else(|_| env::var("LEAP_LIVEKIT_WSS"))
        .map_err(|_| "LIVEKIT_URL or LEAP_LIVEKIT_WSS environment variable is required")?;
    let api_key = env::var("LIVEKIT_API_KEY")
        .or_else(|_| env::var("LEAP_LIVEKIT_API_KEY"))
        .map_err(|_| "LIVEKIT_API_KEY or LEAP_LIVEKIT_API_KEY environment variable is required")?;
    let api_secret = env::var("LIVEKIT_API_SECRET")
        .or_else(|_| env::var("LEAP_LIVEKIT_SECRET_KEY"))
        .map_err(|_| "LIVEKIT_API_SECRET or LEAP_LIVEKIT_SECRET_KEY environment variable is required")?;
    let room_name = env::var("LIVEKIT_ROOM")
        .unwrap_or_else(|_| "video-room".to_string());
    let participant_identity = env::var("LIVEKIT_IDENTITY")
        .unwrap_or_else(|_| "rust-video-renderer".to_string());

    // Generate access token using API key and secret
    let access_token = access_token::AccessToken::with_api_key(&api_key, &api_secret)
        .with_identity(&participant_identity)
        .with_name(&participant_identity)
        .with_grants(access_token::VideoGrants {
            room_join: true,
            room: room_name.clone(),
            ..Default::default()
        })
        .to_jwt()?;

    // Create a minimal runtime for the example
    let runtime = Runtime::new()?;

    // Connect to LiveKit and get video track
    let video_track =
        runtime.block_on(async { create_livekit_video_track(&room_url, &access_token).await })?;

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
            let video_renderer =
                VideoRenderer::new(runtime.handle(), wgpu_render_state.clone(), video_track);

            Box::new(VideoRendererApp { 
                video_renderer,
                brightness: 1.0,
                contrast: 1.0,
                zoom: 1.0,
            })
        }),
    )?;

    Ok(())
}

struct VideoRendererApp {
    video_renderer: VideoRenderer,
    brightness: f32,
    contrast: f32,
    zoom: f32,
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
                let (rect, _response) =
                    ui.allocate_exact_size(placeholder_size, egui::Sense::hover());
                ui.painter()
                    .rect_filled(rect, 0.0, egui::Color32::from_gray(40));

                // Add "No Video" text
                let text_layout = egui::TextLayout::from_galley(
                    ui.fonts(|f| f.layout_single_line("No Video Signal", f32::INFINITY, 24.0)),
                    egui::Color32::from_gray(200),
                    egui::Align2::CENTER_CENTER,
                );
                ui.painter()
                    .add(egui::Shape::text(text_layout, rect.center()));
            }

            ui.add_space(20.0);
            ui.label("The video renderer displays frames from a WebRTC video track.");
            ui.label(
                "In a real application, this would show a live video stream from a LiveKit room.",
            );

            // Add some interactive controls as an example
            ui.add_space(10.0);
            ui.separator();
            ui.add_space(10.0);

            ui.horizontal(|ui| {
                ui.label("Brightness:");
                ui.add(egui::Slider::new(&mut self.brightness, 0.5..=1.5).step_by(0.1));
            });

            ui.horizontal(|ui| {
                ui.label("Contrast:");
                ui.add(egui::Slider::new(&mut self.contrast, 0.5..=1.5).step_by(0.1));
            });

            ui.horizontal(|ui| {
                ui.label("Zoom:");
                ui.add(egui::Slider::new(&mut self.zoom, 1.0..=3.0).step_by(0.1));
            });

            // Display current video statistics
            let (frame_count, dropped_frames, fps) = self.video_renderer.stats();
            ui.add_space(10.0);
            ui.label(format!("Video Statistics:"));
            ui.label(format!("  Frames: {}, Dropped: {}, FPS: {:.1}", frame_count, dropped_frames, fps));
            ui.label(format!("  Brightness: {:.1}, Contrast: {:.1}, Zoom: {:.1}x", self.brightness, self.contrast, self.zoom));

            // Note: These controls don't actually modify the video in this example
            // In a real implementation, you would apply these transformations to the video
        });

        // Request continuous redraw to update the video
        ctx.request_repaint();
    }
}
