use crate::{AudioVisualizer, VideoRenderer};
use egui_wgpu::RenderState;
use fluent_voice_livekit::{ConnectionState, RemoteTrack, Room, RoomEvent};
use livekit_api::access_token;
use std::{collections::HashMap, env, sync::Arc, time::Duration};
use tokio::sync::mpsc;

pub struct LiveKitRoomVisualizer {
    room: Room,
    audio_visualizers: HashMap<String, AudioVisualizer>,
    video_renderers: HashMap<String, VideoRenderer>,
    event_receiver: mpsc::UnboundedReceiver<RoomEvent>,
}

impl LiveKitRoomVisualizer {
    pub async fn new_from_env() -> Result<Self, Box<dyn std::error::Error>> {
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
            .unwrap_or_else(|_| "visualizer-room".to_string());
        let participant_identity = env::var("LIVEKIT_IDENTITY")
            .unwrap_or_else(|_| "rust-room-visualizer".to_string());

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

        Self::new(&room_url, &access_token).await
    }

    pub async fn new(
        room_url: &str,
        access_token: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (room, event_receiver) = tokio::time::timeout(
            Duration::from_secs(10),
            Room::connect(room_url.to_string(), access_token.to_string())
        ).await??;

        Ok(Self {
            room,
            audio_visualizers: HashMap::new(),
            video_renderers: HashMap::new(),
            event_receiver,
        })
    }

    pub async fn process_events(&mut self, rt_handle: &tokio::runtime::Handle, render_state: RenderState) {
        while let Some(event) = self.event_receiver.recv().await {
            match event {
                RoomEvent::TrackSubscribed {
                    track, participant, ..
                } => {
                    let participant_id = participant.identity().0.clone();

                    match track {
                        RemoteTrack::Audio(audio) => {
                            let visualizer = AudioVisualizer::new(rt_handle, audio.0.clone());
                            self.audio_visualizers.insert(participant_id, visualizer);
                        }
                        RemoteTrack::Video(video) => {
                            // Create video renderer with full production implementation
                            let video_renderer = VideoRenderer::new(
                                rt_handle,
                                render_state.clone(),
                                video.0.clone(),
                            );
                            self.video_renderers.insert(participant_id, video_renderer);
                            tracing::info!("Created video renderer for participant");
                        }
                    }
                }
                RoomEvent::TrackUnsubscribed {
                    track, participant, ..
                } => {
                    let participant_id = participant.identity().0.clone();
                    match track {
                        RemoteTrack::Audio(_) => {
                            self.audio_visualizers.remove(&participant_id);
                        }
                        RemoteTrack::Video(_) => {
                            self.video_renderers.remove(&participant_id);
                        }
                    }
                }
                RoomEvent::Disconnected { reason } => {
                    tracing::warn!("Disconnected from room: {}", reason);
                    break;
                }
                _ => {}
            }
        }
    }
}
