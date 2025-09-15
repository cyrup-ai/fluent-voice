use anyhow::Result;
use eframe::egui;
use fluent_voice_animator::{AudioVisualizer, VideoRenderer};
use egui_wgpu::RenderState;
use fluent_voice_livekit::{
    livekit_client::ParticipantIdentity, RemoteTrack, Room, RoomEvent, TrackPublication,
};
use livekit_api::access_token;
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Debug)]
enum ConnectionResult {
    Success { room: Arc<Room>, events_rx: mpsc::UnboundedReceiver<RoomEvent> },
    Error(anyhow::Error),
}

/// Comprehensive LiveKit room visualizer combining audio waveforms, video rendering,
/// and participant management in a single unified interface
struct FullRoomVisualizerApp {
    room: Option<Arc<Room>>,
    room_events_rx: Option<mpsc::UnboundedReceiver<RoomEvent>>,
    connection_rx: Option<mpsc::UnboundedReceiver<ConnectionResult>>,
    rt_handle: tokio::runtime::Handle,
    render_state: Option<RenderState>,
    
    // Participant management
    participants: HashMap<ParticipantIdentity, ParticipantState>,
    selected_participant: Option<ParticipantIdentity>,
    
    // Audio visualization
    audio_visualizers: HashMap<ParticipantIdentity, AudioVisualizer>,
    master_volume: f32,
    
    // Video rendering
    video_renderers: HashMap<ParticipantIdentity, VideoRenderer>,
    
    // UI state
    sidebar_width: f32,
    show_audio_panel: bool,
    show_video_panel: bool,
    show_controls: bool,
    connection_status: String,
    
    // Connection parameters
    room_url: String,
    api_key: String,
    api_secret: String,
    room_name: String,
    participant_name: String,
}

#[derive(Clone, Debug)]
struct ParticipantState {
    identity: ParticipantIdentity,
    name: String,
    audio_track: Option<RemoteTrack>,
    video_track: Option<RemoteTrack>,
    audio_publication: Option<TrackPublication>,
    video_publication: Option<TrackPublication>,
    is_speaking: bool,
    audio_level: f32,
    connection_quality: String,
}

impl Default for FullRoomVisualizerApp {
    fn default() -> Self {
        let rt_handle = tokio::runtime::Handle::current();
        
        Self {
            room: None,
            room_events_rx: None,
            connection_rx: None,
            rt_handle,
            render_state: None,
            participants: HashMap::new(),
            selected_participant: None,
            audio_visualizers: HashMap::new(),
            master_volume: 1.0,
            video_renderers: HashMap::new(),
            sidebar_width: 250.0,
            show_audio_panel: true,
            show_video_panel: true,
            show_controls: true,
            connection_status: "Disconnected".to_string(),
            room_url: env::var("LEAP_LIVEKIT_URL").unwrap_or_default(),
            api_key: env::var("LEAP_LIVEKIT_API_KEY").unwrap_or_default(),
            api_secret: env::var("LEAP_LIVEKIT_API_SECRET").unwrap_or_default(),
            room_name: "fluent-voice-demo".to_string(),
            participant_name: "Rust Visualizer".to_string(),
        }
    }
}

impl eframe::App for FullRoomVisualizerApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Initialize render state if not done
        if self.render_state.is_none() {
            if let Some(wgpu_render_state) = frame.wgpu_render_state() {
                self.render_state = Some(RenderState::new(wgpu_render_state.clone()));
            }
        }

        // Process connection results
        self.process_connection_results();
        
        // Process room events
        self.process_room_events();
        
        // Update participant states with real audio data
        self.update_participant_states();
        
        // Request continuous repaints for real-time updates
        ctx.request_repaint();

        // Main UI layout
        egui::SidePanel::left("participants_panel")
            .resizable(true)
            .default_width(self.sidebar_width)
            .width_range(200.0..=400.0)
            .show(ctx, |ui| {
                self.draw_participants_panel(ui);
            });

        egui::TopBottomPanel::top("controls_panel")
            .resizable(false)
            .show_animated(ctx, self.show_controls, |ui| {
                self.draw_controls_panel(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_main_content(ui);
        });

        // Status bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("Status: {}", self.connection_status));
                ui.separator();
                ui.label(format!("Participants: {}", self.participants.len()));
                ui.separator();
                ui.label(format!("Selected: {}", 
                    self.selected_participant.as_ref()
                        .map(|p| p.0.as_str())
                        .unwrap_or("None")));
            });
        });
    }
}

impl FullRoomVisualizerApp {
    fn process_connection_results(&mut self) {
        if let Some(rx) = &mut self.connection_rx {
            if let Ok(result) = rx.try_recv() {
                match result {
                    ConnectionResult::Success { room, events_rx } => {
                        self.room = Some(room);
                        self.room_events_rx = Some(events_rx);
                        self.connection_status = "Connected".to_string();
                        println!("Successfully connected to room via channel");
                    }
                    ConnectionResult::Error(e) => {
                        self.connection_status = format!("Connection failed: {:?}", e);
                        println!("Failed to connect to room: {:?}", e);
                    }
                }
            }
        }
    }
    
    fn update_participant_states(&mut self) {
        for (participant_id, audio_viz) in &self.audio_visualizers {
            if let Some(participant_state) = self.participants.get_mut(participant_id) {
                let stats = audio_viz.get_stats();
                participant_state.audio_level = stats.current_amplitude;
                participant_state.is_speaking = stats.current_amplitude > 0.01; // Threshold for speaking detection
            }
        }
    }
    
    fn process_room_events(&mut self) {
        if let Some(rx) = &mut self.room_events_rx {
            while let Ok(event) = rx.try_recv() {
                match event {
                    RoomEvent::ParticipantConnected(participant) => {
                        let identity = participant.identity();
                        let name = format!("Participant {}", identity.0);
                        
                        self.participants.insert(identity.clone(), ParticipantState {
                            identity: identity.clone(),
                            name,
                            audio_track: None,
                            video_track: None,
                            audio_publication: None,
                            video_publication: None,
                            is_speaking: false,
                            audio_level: 0.0,
                            connection_quality: "Good".to_string(),
                        });
                        
                        println!("Participant connected: {}", identity.0);
                    }
                    
                    RoomEvent::ParticipantDisconnected(participant) => {
                        let identity = participant.identity();
                        self.participants.remove(&identity);
                        self.audio_visualizers.remove(&identity);
                        self.video_renderers.remove(&identity);
                        
                        if self.selected_participant.as_ref() == Some(&identity) {
                            self.selected_participant = None;
                        }
                        
                        println!("Participant disconnected: {}", identity.0);
                    }
                    
                    RoomEvent::TrackSubscribed { track, participant, publication } => {
                        let participant_id = participant.identity();
                        
                        if let Some(participant_state) = self.participants.get_mut(&participant_id) {
                            match track {
                                RemoteTrack::Audio(audio_track) => {
                                    participant_state.audio_track = Some(track.clone());
                                    participant_state.audio_publication = Some(publication);
                                    
                                    // Create audio visualizer for this participant
                                    let audio_visualizer = AudioVisualizer::new(
                                        self.rt_handle.clone(),
                                        audio_track.0.clone(),
                                    );
                                    self.audio_visualizers.insert(participant_id.clone(), audio_visualizer);
                                    
                                    println!("Audio track subscribed for: {}", participant_id.0);
                                }
                                RemoteTrack::Video(video_track) => {
                                    participant_state.video_track = Some(track.clone());
                                    participant_state.video_publication = Some(publication);
                                    
                                    // Create video renderer for this participant
                                    if let Some(render_state) = &self.render_state {
                                        let video_renderer = VideoRenderer::new(
                                            &self.rt_handle,
                                            render_state.clone(),
                                            video_track.0.clone(),
                                        );
                                        self.video_renderers.insert(participant_id.clone(), video_renderer);
                                    }
                                    
                                    println!("Video track subscribed for: {}", participant_id.0);
                                }
                            }
                        }
                    }
                    
                    RoomEvent::TrackUnsubscribed { participant, .. } => {
                        let participant_id = participant.identity();
                        
                        if let Some(participant_state) = self.participants.get_mut(&participant_id) {
                            participant_state.audio_track = None;
                            participant_state.video_track = None;
                            participant_state.audio_publication = None;
                            participant_state.video_publication = None;
                        }
                        
                        self.audio_visualizers.remove(&participant_id);
                        self.video_renderers.remove(&participant_id);
                        
                        println!("Track unsubscribed for: {}", participant_id.0);
                    }
                    
                    RoomEvent::Connected { participants_with_tracks } => {
                        self.connection_status = "Connected".to_string();
                        
                        for (participant, publications) in participants_with_tracks {
                            let identity = participant.identity();
                            let name = format!("Participant {}", identity.0);
                            
                            self.participants.insert(identity.clone(), ParticipantState {
                                identity: identity.clone(),
                                name,
                                audio_track: None,
                                video_track: None,
                                audio_publication: None,
                                video_publication: None,
                                is_speaking: false,
                                audio_level: 0.0,
                                connection_quality: "Good".to_string(),
                            });
                        }
                        
                        println!("Connected to room with {} participants", self.participants.len());
                    }
                    
                    RoomEvent::Disconnected { reason } => {
                        self.connection_status = format!("Disconnected: {}", reason);
                        self.participants.clear();
                        self.audio_visualizers.clear();
                        self.video_renderers.clear();
                        self.selected_participant = None;
                        
                        println!("Disconnected from room: {}", reason);
                    }
                    
                    _ => {
                        // Handle other events as needed
                    }
                }
            }
        }
    }

    fn draw_participants_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Participants");
        
        ui.separator();
        
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (participant_id, participant_state) in &self.participants {
                let is_selected = self.selected_participant.as_ref() == Some(participant_id);
                
                let response = ui.selectable_label(is_selected, &participant_state.name);
                
                if response.clicked() {
                    self.selected_participant = Some(participant_id.clone());
                }
                
                // Show participant status indicators
                ui.horizontal(|ui| {
                    if participant_state.audio_track.is_some() {
                        ui.colored_label(egui::Color32::GREEN, "🎤");
                    }
                    if participant_state.video_track.is_some() {
                        ui.colored_label(egui::Color32::BLUE, "📹");
                    }
                    if participant_state.is_speaking {
                        ui.colored_label(egui::Color32::YELLOW, "🔊");
                    }
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.small(format!("{}%", (participant_state.audio_level * 100.0) as u32));
                    });
                });
                
                ui.separator();
            }
        });
        
        ui.separator();
        
        // Connection controls
        if self.room.is_none() {
            ui.heading("Connect to Room");
            
            ui.horizontal(|ui| {
                ui.label("Room URL:");
                ui.text_edit_singleline(&mut self.room_url);
            });
            
            ui.horizontal(|ui| {
                ui.label("API Key:");
                ui.text_edit_singleline(&mut self.api_key);
            });
            
            ui.horizontal(|ui| {
                ui.label("API Secret:");
                ui.text_edit_singleline(&mut self.api_secret);
            });
            
            ui.horizontal(|ui| {
                ui.label("Room Name:");
                ui.text_edit_singleline(&mut self.room_name);
            });
            
            ui.horizontal(|ui| {
                ui.label("Your Name:");
                ui.text_edit_singleline(&mut self.participant_name);
            });
            
            if ui.button("Connect").clicked() {
                self.connect_to_room();
            }
        } else {
            if ui.button("Disconnect").clicked() {
                self.disconnect_from_room();
            }
        }
    }

    fn draw_controls_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_audio_panel, "Audio Panel");
            ui.checkbox(&mut self.show_video_panel, "Video Panel");
            
            ui.separator();
            
            ui.label("Master Volume:");
            let old_volume = self.master_volume;
            ui.add(egui::Slider::new(&mut self.master_volume, 0.0..=2.0).suffix("%"));
            
            // Apply volume changes to audio visualizers
            if (self.master_volume - old_volume).abs() > 0.01 {
                for audio_viz in self.audio_visualizers.values_mut() {
                    // Note: Volume control may require extending AudioVisualizer
                    // or applying at the audio stream level - this is a placeholder
                    // for where volume would be applied to the audio system
                }
            }
            
            ui.separator();
            
            if ui.button("🔄 Refresh").clicked() {
                // Refresh participant list or reconnect
                println!("Refreshing room state...");
            }
        });
    }

    fn draw_main_content(&mut self, ui: &mut egui::Ui) {
        if self.show_audio_panel && self.show_video_panel {
            // Split view: audio on left, video on right
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.set_width(ui.available_width() * 0.5);
                    self.draw_audio_panel(ui);
                });
                
                ui.separator();
                
                ui.vertical(|ui| {
                    self.draw_video_panel(ui);
                });
            });
        } else if self.show_audio_panel {
            self.draw_audio_panel(ui);
        } else if self.show_video_panel {
            self.draw_video_panel(ui);
        } else {
            ui.centered_and_justified(|ui| {
                ui.heading("Select a panel to view content");
            });
        }
    }

    fn draw_audio_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Audio Visualization");
        
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (participant_id, visualizer) in &self.audio_visualizers {
                if let Some(participant_state) = self.participants.get(participant_id) {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.heading(&participant_state.name);
                            
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if participant_state.is_speaking {
                                    ui.colored_label(egui::Color32::GREEN, "Speaking");
                                }
                            });
                        });
                        
                        // Draw waveform visualization
                        let response = ui.allocate_response(
                            egui::Vec2::new(ui.available_width(), 100.0),
                            egui::Sense::hover()
                        );
                        
                        if let Some(painter) = ui.ctx().try_get_painter() {
                            visualizer.paint_waveform(&painter, response.rect);
                        }
                        
                        // Audio controls
                        ui.horizontal(|ui| {
                            ui.label("Volume:");
                            let mut volume = participant_state.audio_level;
                            ui.add(egui::Slider::new(&mut volume, 0.0..=1.0).suffix("%"));
                            
                            if ui.button("Mute").clicked() {
                                // Connect to actual LiveKit mute APIs
                                if let Some(track_pub) = &participant_state.audio_publication {
                                    match track_pub {
                                        TrackPublication::Local(local_pub) => {
                                            if local_pub.is_muted() {
                                                local_pub.unmute();
                                            } else {
                                                local_pub.mute();
                                            }
                                        }
                                        TrackPublication::Remote(remote_pub) => {
                                            remote_pub.set_enabled(!remote_pub.is_enabled());
                                        }
                                    }
                                }
                                println!("Toggling mute for {}", participant_id.0);
                            }
                        });
                    });
                    
                    ui.add_space(10.0);
                }
            }
        });
    }

    fn draw_video_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Video Display");
        
        if let Some(selected_id) = &self.selected_participant {
            if let Some(video_renderer) = self.video_renderers.get(selected_id) {
                if let Some(participant_state) = self.participants.get(selected_id) {
                    ui.horizontal(|ui| {
                        ui.heading(format!("Video: {}", participant_state.name));
                        
                        let (frame_count, dropped_frames, fps) = video_renderer.stats();
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.small(format!("FPS: {:.1}", fps));
                            ui.small(format!("Dropped: {}", dropped_frames));
                            ui.small(format!("Frames: {}", frame_count));
                        });
                    });
                    
                    // Video display area
                    let available_size = ui.available_size();
                    let video_response = ui.allocate_response(available_size, egui::Sense::hover());
                    
                    if let Some(texture_id) = video_renderer.texture_id() {
                        let (width, height) = video_renderer.resolution();
                        
                        if width > 0 && height > 0 {
                            // Calculate aspect ratio and fit video in available space
                            let video_aspect = width as f32 / height as f32;
                            let available_aspect = available_size.x / available_size.y;
                            
                            let (display_width, display_height) = if video_aspect > available_aspect {
                                (available_size.x, available_size.x / video_aspect)
                            } else {
                                (available_size.y * video_aspect, available_size.y)
                            };
                            
                            let display_size = egui::Vec2::new(display_width, display_height);
                            let display_rect = egui::Rect::from_center_size(
                                video_response.rect.center(),
                                display_size
                            );
                            
                            ui.allocate_ui_at_rect(display_rect, |ui| {
                                ui.image((texture_id, display_size));
                            });
                        }
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.label("Waiting for video...");
                        });
                    }
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Selected participant has no video track");
                });
            }
        } else {
            ui.centered_and_justified(|ui| {
                ui.label("Select a participant to view video");
            });
        }
    }

    fn connect_to_room(&mut self) {
        if self.room_url.is_empty() || self.api_key.is_empty() || self.api_secret.is_empty() {
            println!("Missing connection parameters");
            return;
        }
        
        let rt_handle = self.rt_handle.clone();
        let url = self.room_url.clone();
        let api_key = self.api_key.clone();
        let api_secret = self.api_secret.clone();
        let room_name = self.room_name.clone();
        let participant_name = self.participant_name.clone();
        
        // Create channel for connection results
        let (connection_tx, connection_rx) = mpsc::unbounded_channel();
        self.connection_rx = Some(connection_rx);
        
        // Spawn connection task
        rt_handle.spawn(async move {
            match Self::establish_room_connection(url, api_key, api_secret, room_name, participant_name).await {
                Ok((room, events_rx)) => {
                    let _ = connection_tx.send(ConnectionResult::Success { room, events_rx });
                }
                Err(e) => {
                    let _ = connection_tx.send(ConnectionResult::Error(e));
                }
            }
        });
    }
    
    async fn establish_room_connection(
        url: String,
        api_key: String,
        api_secret: String,
        room_name: String,
        participant_name: String,
    ) -> Result<(Arc<Room>, mpsc::UnboundedReceiver<RoomEvent>)> {
        // Generate JWT token
        let token = access_token::AccessToken::with_api_key(&api_key, &api_secret)
            .with_identity(&participant_name)
            .with_name(&participant_name)
            .with_grants(access_token::VideoGrants {
                room_join: true,
                room: room_name,
                ..Default::default()
            })
            .to_jwt()?;
        
        // Connect to room
        let (room, events_rx) = Room::connect(url, token).await?;
        
        Ok((Arc::new(room), events_rx))
    }

    fn disconnect_from_room(&mut self) {
        self.room = None;
        self.room_events_rx = None;
        self.connection_rx = None;
        self.participants.clear();
        self.audio_visualizers.clear();
        self.video_renderers.clear();
        self.selected_participant = None;
        self.connection_status = "Disconnected".to_string();
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0]),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            supported_backends: eframe::wgpu::Backends::PRIMARY,
            device_descriptor: Arc::new(|_| eframe::wgpu::DeviceDescriptor {
                label: Some("egui wgpu device"),
                required_features: eframe::wgpu::Features::default(),
                required_limits: eframe::wgpu::Limits::default(),
                memory_hints: eframe::wgpu::MemoryHints::default(),
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "LiveKit Full Room Visualizer",
        options,
        Box::new(|_cc| {
            let app = FullRoomVisualizerApp::default();
            Ok(Box::new(app))
        }),
    )?;

    Ok(())
}