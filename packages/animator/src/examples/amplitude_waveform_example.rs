use crate::AudioVisualizer;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use fluent_voice_livekit::{RemoteAudioTrack, RemoteTrack, Room, RoomEvent};
use futures::StreamExt;
use livekit::webrtc::prelude::*;
use livekit_api::access_token;
use std::{env, sync::Arc, time::Duration};
use tokio::runtime::Runtime;

// Replace mock function with real LiveKit connection
async fn create_livekit_audio_track(
    room_url: &str,
    access_token: &str,
) -> Result<livekit::webrtc::prelude::RtcAudioTrack, Box<dyn std::error::Error>> {
    // Connect to LiveKit room with timeout
    let (room, mut events) = tokio::time::timeout(
        Duration::from_secs(10),
        Room::connect(room_url.to_string(), access_token.to_string())
    ).await??;

    // Wait for first audio track from any participant with timeout
    let timeout_duration = Duration::from_secs(30);
    let start_time = std::time::Instant::now();
    
    while let Some(event) = events.next().await {
        if start_time.elapsed() > timeout_duration {
            return Err("Timeout waiting for audio track".into());
        }
        match event {
            RoomEvent::TrackSubscribed {
                track: RemoteTrack::Audio(audio_track),
                publication,
                participant,
            } => {
                tracing::info!(
                    participant = %participant.identity().0,
                    track_sid = %audio_track.sid(),
                    "Connected to remote audio track"
                );

                // Enable the track
                publication.set_enabled(true);

                // Return the underlying RTC track for AudioVisualizer
                return Ok(audio_track.0.clone());
            }
            RoomEvent::Connected {
                participants_with_tracks,
            } => {
                // Check existing participants for audio tracks
                for (participant, publications) in participants_with_tracks {
                    for publication in publications {
                        if publication.is_audio() {
                            if let Some(RemoteTrack::Audio(track)) = publication.track() {
                                tracing::info!(
                                    participant = %participant.identity().0,
                                    "Found existing audio track"
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

    Err("No audio track found in room".into())
}

// Update the example runner to handle async and connection params
pub fn run_amplitude_waveform_example() -> Result<(), Box<dyn std::error::Error>> {
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
        .unwrap_or_else(|_| "audio-room".to_string());
    let participant_identity = env::var("LIVEKIT_IDENTITY")
        .unwrap_or_else(|_| "rust-audio-visualizer".to_string());

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

    // Create tokio runtime for async operations
    let runtime = Runtime::new()?;

    // Connect to LiveKit and get audio track
    let audio_track = runtime.block_on(async { 
        create_livekit_audio_track(&room_url, &access_token).await 
    })?;

    // Create the audio visualizer
    let visualizer = AudioVisualizer::new(runtime.handle(), audio_track);

    // Create a simple egui application to display the visualization
    let native_options = eframe::NativeOptions {
        initial_window_size: Some(Vec2::new(800.0, 400.0)),
        ..Default::default()
    };

    eframe::run_native(
        "Audio Amplitude Visualization Example",
        native_options,
        Box::new(|_cc| Box::new(AmplitudeVisApp { visualizer })),
    )?;

    Ok(())
}

struct AmplitudeVisApp {
    visualizer: AudioVisualizer,
}

impl eframe::App for AmplitudeVisApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Audio Amplitude Waveform");
            ui.add_space(20.0);
            // Get the latest amplitude data
            let amplitudes = self.visualizer.get_amplitudes();

            // Create a waveform visualization
            let available_width = ui.available_width();
            let height = 200.0;
            let rect = ui.allocate_space(Vec2::new(available_width, height)).0;
            
            if !amplitudes.is_empty() {
                let painter = ui.painter();
                
                // Draw the baseline
                let baseline_y = rect.min.y + height / 2.0;
                painter.line_segment(
                    [Pos2::new(rect.min.x, baseline_y), Pos2::new(rect.max.x, baseline_y)],
                    Stroke::new(1.0, Color32::from_gray(100)),
                );
                
                // Draw the waveform
                let points_per_segment = amplitudes.len().max(1);
                let segment_width = available_width / points_per_segment as f32;
                
                // Scale factor for visualization (adjust as needed)
                let amplitude_scale = height / 2.0;
                
                // Draw waveform lines
                for i in 0..amplitudes.len().saturating_sub(1) {
                    let x1 = rect.min.x + i as f32 * segment_width;
                    let x2 = rect.min.x + (i + 1) as f32 * segment_width;
                    
                    let y1 = baseline_y - amplitudes[i] * amplitude_scale;
                    let y2 = baseline_y - amplitudes[i + 1] * amplitude_scale;
                    
                    painter.line_segment(
                        [Pos2::new(x1, y1), Pos2::new(x2, y2)],
                        Stroke::new(2.0, Color32::from_rgb(30, 144, 255)), // DodgerBlue
                    );
                }
                
                // Draw the envelope
                let mut envelope_points = Vec::with_capacity(amplitudes.len() * 2);
                
                // Add points for the upper part of the envelope
                for i in 0..amplitudes.len() {
                    let x = rect.min.x + i as f32 * segment_width;
                    let y = baseline_y - amplitudes[i] * amplitude_scale;
                    envelope_points.push(Pos2::new(x, y));
                }
                
                // Add points for the lower part of the envelope (in reverse)
                for i in (0..amplitudes.len()).rev() {
                    let x = rect.min.x + i as f32 * segment_width;
                    let y = baseline_y + amplitudes[i] * amplitude_scale;
                    envelope_points.push(Pos2::new(x, y));
                }
                
                // Draw the filled envelope
                painter.add(egui::Shape::convex_polygon(
                    envelope_points,
                    Color32::from_rgba_premultiplied(30, 144, 255, 40), // Semi-transparent blue
                    Stroke::new(1.0, Color32::from_rgb(30, 144, 255)),
                ));
            }
            
            ui.add_space(20.0);
            ui.label("The visualization shows the audio amplitude over time, with the most recent values on the right.");
            ui.label("The blue area represents the envelope of the audio signal.");
        });

        // Request continuous redraw to update the visualization
        ctx.request_repaint();
    }
}
