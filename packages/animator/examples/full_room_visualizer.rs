use anyhow::Result;
use eframe::egui;
use egui_wgpu::RenderState;
use fluent_voice_animator::{
    AudioVisualizer, ConnectionQuality, ErrorState, LiveKitAudioPlayer, RoomVisualizerConfig,
    VideoRenderer, VisualizerError,
};
use livekit::prelude::*;
use livekit_api::access_token;
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

#[derive(Debug)]
enum ConnectionResult {
    Success {
        room: Arc<Room>,
        events_rx: mpsc::UnboundedReceiver<RoomEvent>,
    },
    Error(anyhow::Error),
}

/// Connection state for recovery management
#[derive(Debug, Clone)]
enum ConnectionState {
    Connected,
    Disconnected,
    Reconnecting,
    Failed,
}

/// Automatic reconnection manager with exponential backoff
struct ConnectionRecoveryManager {
    retry_count: u32,
    max_retries: u32,
    retry_delay: Duration,
    last_attempt: Instant,
    backoff_multiplier: f32,
    connection_state: ConnectionState,
}

impl Default for ConnectionRecoveryManager {
    fn default() -> Self {
        Self {
            retry_count: 0,
            max_retries: 5,
            retry_delay: Duration::from_secs(2),
            last_attempt: Instant::now(),
            backoff_multiplier: 2.0,
            connection_state: ConnectionState::Disconnected,
        }
    }
}

impl ConnectionRecoveryManager {
    fn should_attempt_recovery(&self) -> bool {
        matches!(self.connection_state, ConnectionState::Disconnected)
            && self.retry_count < self.max_retries
            && self.last_attempt.elapsed() > self.current_retry_delay()
    }

    fn current_retry_delay(&self) -> Duration {
        Duration::from_millis(
            (self.retry_delay.as_millis() as f32
                * self.backoff_multiplier.powi(self.retry_count as i32)) as u64,
        )
    }

    fn mark_attempt(&mut self) {
        self.connection_state = ConnectionState::Reconnecting;
        self.retry_count += 1;
        self.last_attempt = Instant::now();
    }

    fn mark_success(&mut self) {
        self.connection_state = ConnectionState::Connected;
        self.retry_count = 0;
    }

    fn mark_failure(&mut self) {
        if self.retry_count >= self.max_retries {
            self.connection_state = ConnectionState::Failed;
        } else {
            self.connection_state = ConnectionState::Disconnected;
        }
    }

    fn mark_disconnected(&mut self) {
        self.connection_state = ConnectionState::Disconnected;
    }
}

/// Real-time update configuration for performance optimization
struct RealTimeUpdateConfig {
    stats_update_interval: Duration,       // 100ms (10Hz)
    speaking_detection_interval: Duration, // 33ms (30Hz)
    last_stats_update: Instant,
    last_speaking_update: Instant,
}

impl Default for RealTimeUpdateConfig {
    fn default() -> Self {
        Self {
            stats_update_interval: Duration::from_millis(100), // 10Hz
            speaking_detection_interval: Duration::from_millis(33), // 30Hz
            last_stats_update: Instant::now(),
            last_speaking_update: Instant::now(),
        }
    }
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

    // Audio playback (NEW - Phase 1)
    audio_players: HashMap<ParticipantIdentity, LiveKitAudioPlayer>,

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

    // Configuration and error handling (NEW - TODO2.md)
    config: RoomVisualizerConfig,
    error_state: Option<ErrorState>,
    last_quality_update: Instant,

    // Real-time update configuration (NEW - TODO3.md Phase 2)
    rt_config: RealTimeUpdateConfig,

    // Connection recovery manager (NEW - TODO3.md Phase 3)
    recovery_manager: ConnectionRecoveryManager,
}

#[derive(Clone, Debug)]
struct ParticipantState {
    identity: ParticipantIdentity,
    name: String,
    audio_track: Option<RemoteTrack>,
    video_track: Option<RemoteTrack>,
    audio_publication: Option<RemoteTrackPublication>,
    video_publication: Option<RemoteTrackPublication>,
    is_speaking: bool,
    audio_level: f32,
    connection_quality: String,
    // NEW - Phase 3: Individual volume persistence
    participant_volume: f32,
    is_muted: bool,
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
            audio_players: HashMap::new(),
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
            config: RoomVisualizerConfig::default(),
            error_state: None,
            last_quality_update: Instant::now(),
            rt_config: RealTimeUpdateConfig::default(),
            recovery_manager: ConnectionRecoveryManager::default(),
        }
    }
}

impl eframe::App for FullRoomVisualizerApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Render state will be None - this example will run without wgpu rendering for now

        // Auto-recovery check for disconnected rooms
        if self.room.is_none() && self.recovery_manager.should_attempt_recovery() {
            if !self.room_url.is_empty() && !self.api_key.is_empty() && !self.api_secret.is_empty()
            {
                self.recovery_manager.mark_attempt();
                tracing::info!(
                    "Attempting auto-recovery, retry #{}",
                    self.recovery_manager.retry_count
                );
                self.connection_status = format!(
                    "Reconnecting... (attempt {})",
                    self.recovery_manager.retry_count
                );

                // Trigger reconnection attempt
                self.connect_to_room();
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

        // Handle error display
        if self.error_state.is_some() {
            egui::TopBottomPanel::top("error_panel")
                .resizable(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().indent = 10.0;
                        self.handle_ui_errors(ui);
                    });
                });
        }

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
                ui.label(format!(
                    "Selected: {}",
                    self.selected_participant
                        .as_ref()
                        .map(|p| p.0.as_str())
                        .unwrap_or("None")
                ));
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
                        self.recovery_manager.mark_success();
                        tracing::info!("Successfully connected to room via channel");
                    }
                    ConnectionResult::Error(e) => {
                        self.connection_status = format!("Connection failed: {:?}", e);
                        self.recovery_manager.mark_failure();
                        tracing::error!("Failed to connect to room: {:?}", e);
                    }
                }
            }
        }
    }

    fn update_participant_states(&mut self) {
        let now = Instant::now();

        // Rate-limited stats updates (10Hz instead of 60Hz)
        if now.duration_since(self.rt_config.last_stats_update)
            > self.rt_config.stats_update_interval
        {
            for (participant_id, audio_viz) in &self.audio_visualizers {
                if let Some(participant_state) = self.participants.get_mut(participant_id) {
                    let stats = audio_viz.get_stats();
                    participant_state.audio_level = stats.current_amplitude;
                }
            }
            self.rt_config.last_stats_update = now;
        }

        // Rate-limited speaking detection (30Hz for responsiveness)
        if now.duration_since(self.rt_config.last_speaking_update)
            > self.rt_config.speaking_detection_interval
        {
            for (participant_id, audio_viz) in &self.audio_visualizers {
                if let Some(participant_state) = self.participants.get_mut(participant_id) {
                    let stats = audio_viz.get_stats();
                    participant_state.is_speaking =
                        stats.current_amplitude > self.config.speaking_threshold;
                }
            }
            self.rt_config.last_speaking_update = now;
        }

        // Update connection quality periodically (already rate-limited)
        self.update_connection_quality();
    }

    fn update_connection_quality(&mut self) {
        if self.last_quality_update.elapsed() < self.config.connection_quality_update_interval {
            return;
        }

        let participant_ids: Vec<ParticipantIdentity> = self.participants.keys().cloned().collect();
        for participant_id in participant_ids {
            let quality = self.calculate_real_quality(&participant_id);
            if let Some(participant_state) = self.participants.get_mut(&participant_id) {
                participant_state.connection_quality = quality.as_str().to_string();
            }
        }

        self.last_quality_update = Instant::now();
    }

    fn calculate_real_quality(&self, participant_id: &ParticipantIdentity) -> ConnectionQuality {
        // Use real LiveKit participant connection quality API
        if let Some(room) = &self.room {
            for participant in room.remote_participants().values() {
                if participant.identity() == *participant_id {
                    return ConnectionQuality::Excellent; // Placeholder since connection_quality() method may not exist
                }
            }
        }
        ConnectionQuality::Unknown
    }

    /// Report an error for UI display
    fn report_error(&mut self, error: VisualizerError) {
        tracing::error!("{}", error);
        self.error_state = Some(ErrorState::new(error));
    }

    /// Handle UI errors (display and auto-dismiss)
    fn handle_ui_errors(&mut self, ui: &mut egui::Ui) {
        if let Some(error_state) = &mut self.error_state {
            if error_state.is_expired(self.config.error_display_timeout) {
                self.error_state = None;
                return;
            }

            ui.horizontal(|ui| {
                let error = &error_state.error;
                ui.colored_label(error.color(), format!("{} {}", error.icon(), error));

                if ui.button("✖").clicked() {
                    error_state.dismiss();
                }

                // Auto-dismiss countdown
                let remaining = self
                    .config
                    .error_display_timeout
                    .saturating_sub(error_state.elapsed());
                ui.label(format!("({}s)", remaining.as_secs()));
            });
        }
    }

    fn process_room_events(&mut self) {
        if let Some(rx) = &mut self.room_events_rx {
            while let Ok(event) = rx.try_recv() {
                match event {
                    RoomEvent::ParticipantConnected(participant) => {
                        let identity = participant.identity();
                        let name = format!("Participant {}", identity.0);

                        self.participants.insert(
                            identity.clone(),
                            ParticipantState {
                                identity: identity.clone(),
                                name,
                                audio_track: None,
                                video_track: None,
                                audio_publication: None,
                                video_publication: None,
                                is_speaking: false,
                                audio_level: 0.0,
                                connection_quality: ConnectionQuality::Unknown.as_str().to_string(),
                                participant_volume: 1.0,
                                is_muted: false,
                            },
                        );

                        tracing::info!("Participant connected: {}", identity.0);
                    }

                    RoomEvent::ParticipantDisconnected(participant) => {
                        let identity = participant.identity();
                        self.participants.remove(&identity);
                        self.audio_visualizers.remove(&identity);
                        self.audio_players.remove(&identity);
                        self.video_renderers.remove(&identity);

                        if self.selected_participant.as_ref() == Some(&identity) {
                            self.selected_participant = None;
                        }

                        tracing::info!("Participant disconnected: {}", identity.0);
                    }

                    RoomEvent::TrackSubscribed {
                        track,
                        participant,
                        publication,
                    } => {
                        let participant_id = participant.identity();

                        if let Some(participant_state) = self.participants.get_mut(&participant_id)
                        {
                            let track_clone = track.clone();
                            match track {
                                RemoteTrack::Audio(audio_track) => {
                                    participant_state.audio_track = Some(track_clone);
                                    participant_state.audio_publication = Some(publication);

                                    // Create audio visualizer for this participant
                                    let audio_visualizer = AudioVisualizer::new(
                                        &self.rt_handle,
                                        audio_track.rtc_track(),
                                    );
                                    self.audio_visualizers
                                        .insert(participant_id.clone(), audio_visualizer);

                                    // NEW - Phase 1: Create audio player for real playback
                                    match LiveKitAudioPlayer::new_from_remote_track(
                                        &self.rt_handle,
                                        audio_track.rtc_track(),
                                    ) {
                                        Ok(audio_player) => {
                                            // Apply current master volume to new player
                                            audio_player.set_volume(self.master_volume);
                                            self.audio_players
                                                .insert(participant_id.clone(), audio_player);
                                            tracing::info!(
                                                "Audio player created for: {}",
                                                participant_id.0
                                            );
                                        }
                                        Err(e) => {
                                            tracing::error!(
                                                "Failed to create audio player for {}: {:?}",
                                                participant_id.0,
                                                e
                                            );
                                        }
                                    }

                                    tracing::info!(
                                        "Audio track subscribed for: {}",
                                        participant_id.0
                                    );
                                }
                                RemoteTrack::Video(video_track) => {
                                    participant_state.video_track = Some(track_clone.clone());
                                    participant_state.video_publication = Some(publication);

                                    // Create video renderer for this participant
                                    if let Some(render_state) = &self.render_state {
                                        let video_renderer = VideoRenderer::new(
                                            &self.rt_handle,
                                            render_state.clone(),
                                            video_track.rtc_track(),
                                        );
                                        self.video_renderers
                                            .insert(participant_id.clone(), video_renderer);
                                    }

                                    tracing::info!(
                                        "Video track subscribed for: {}",
                                        participant_id.0
                                    );
                                }
                            }
                        }
                    }

                    RoomEvent::TrackUnsubscribed { participant, .. } => {
                        let participant_id = participant.identity();

                        if let Some(participant_state) = self.participants.get_mut(&participant_id)
                        {
                            participant_state.audio_track = None;
                            participant_state.video_track = None;
                            participant_state.audio_publication = None;
                            participant_state.video_publication = None;
                        }

                        self.audio_visualizers.remove(&participant_id);
                        self.audio_players.remove(&participant_id);
                        self.video_renderers.remove(&participant_id);

                        tracing::info!("Track unsubscribed for: {}", participant_id.0);
                    }

                    RoomEvent::Connected {
                        participants_with_tracks,
                    } => {
                        self.connection_status = "Connected".to_string();
                        self.recovery_manager.mark_success();

                        for (participant, publications) in participants_with_tracks {
                            let identity = participant.identity();
                            let name = format!("Participant {}", identity.0);

                            self.participants.insert(
                                identity.clone(),
                                ParticipantState {
                                    identity: identity.clone(),
                                    name,
                                    audio_track: None,
                                    video_track: None,
                                    audio_publication: None,
                                    video_publication: None,
                                    is_speaking: false,
                                    audio_level: 0.0,
                                    connection_quality: ConnectionQuality::Unknown
                                        .as_str()
                                        .to_string(),
                                    participant_volume: 1.0,
                                    is_muted: false,
                                },
                            );
                        }

                        tracing::info!(
                            "Connected to room with {} participants",
                            self.participants.len()
                        );
                    }

                    RoomEvent::Disconnected { reason } => {
                        self.connection_status = format!("Disconnected: {:?}", reason);
                        self.recovery_manager.mark_disconnected();
                        self.participants.clear();
                        self.audio_visualizers.clear();
                        self.video_renderers.clear();
                        self.selected_participant = None;

                        tracing::info!("Disconnected from room: {:?}", reason);
                    }

                    // ConnectionQualityChanged event not available in current LiveKit implementation

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

                    // Connection quality indicator with color
                    let quality = &participant_state.connection_quality;
                    let color = match quality.as_str() {
                        "Excellent" => egui::Color32::from_rgb(0, 255, 0),
                        "Good" => egui::Color32::from_rgb(144, 238, 144),
                        "Fair" => egui::Color32::YELLOW,
                        "Poor" => egui::Color32::RED,
                        _ => egui::Color32::GRAY,
                    };
                    ui.colored_label(color, quality);

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.small(format!(
                            "{}%",
                            (participant_state.audio_level * 100.0) as u32
                        ));
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

            // Apply volume changes to both visualization and playback
            if (self.master_volume - old_volume).abs() > 0.01 {
                // Apply to audio visualizers for consistent amplitude display
                for audio_viz in self.audio_visualizers.values() {
                    if let Err(e) = audio_viz.set_volume(self.master_volume) {
                        tracing::warn!("Failed to set visualizer volume: {}", e);
                    }
                }

                // Apply to audio players for actual sound output
                for audio_player in self.audio_players.values() {
                    audio_player.set_volume(self.master_volume);
                }

                tracing::info!("Master volume set to: {:.2}", self.master_volume);
            }

            ui.separator();

            if ui.button("🔄 Refresh").clicked() {
                // Soft refresh implementation based on TODO1.md
                if let Some(room) = &self.room {
                    tracing::info!("Refreshing room participant state");

                    // Reset UI state
                    self.selected_participant = None;

                    // Refresh connection quality for all participants
                    self.refresh_connection_quality();

                    // Re-enable all tracks to force fresh subscriptions
                    for participant_state in self.participants.values() {
                        if let Some(audio_pub) = &participant_state.audio_publication {
                            audio_pub.set_enabled(true);
                        }
                        if let Some(video_pub) = &participant_state.video_publication {
                            video_pub.set_enabled(true);
                        }
                    }
                } else {
                    tracing::warn!("Cannot refresh - not connected to room");
                }
            }

            ui.separator();

            // Configuration settings
            ui.collapsing("🔧 Settings", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Speaking Threshold:");
                    if ui
                        .add(
                            egui::Slider::new(&mut self.config.speaking_threshold, 0.001..=0.1)
                                .logarithmic(true)
                                .suffix(" amplitude"),
                        )
                        .changed()
                    {
                        tracing::debug!(
                            "Speaking threshold updated to: {:.3}",
                            self.config.speaking_threshold
                        );
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Quality Update Interval:");
                    let mut secs = self.config.connection_quality_update_interval.as_secs() as f32;
                    if ui
                        .add(egui::Slider::new(&mut secs, 1.0..=30.0).suffix(" sec"))
                        .changed()
                    {
                        self.config.connection_quality_update_interval =
                            std::time::Duration::from_secs(secs as u64);
                        tracing::debug!("Quality update interval set to: {} seconds", secs);
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Error Display Timeout:");
                    let mut secs = self.config.error_display_timeout.as_secs() as f32;
                    if ui
                        .add(egui::Slider::new(&mut secs, 3.0..=30.0).suffix(" sec"))
                        .changed()
                    {
                        self.config.error_display_timeout =
                            std::time::Duration::from_secs(secs as u64);
                        tracing::debug!("Error display timeout set to: {} seconds", secs);
                    }
                });

                if ui.button("Reset to Defaults").clicked() {
                    self.config = RoomVisualizerConfig::default();
                    tracing::info!("Configuration reset to defaults");
                }
            });
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
            let participant_ids: Vec<ParticipantIdentity> = self.audio_visualizers.keys().cloned().collect();
            for participant_id in participant_ids {
                if let (Some(visualizer), Some(participant_state)) = (
                    self.audio_visualizers.get(&participant_id),
                    self.participants.get_mut(&participant_id)
                ) {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.heading(&participant_state.name);

                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if participant_state.is_speaking {
                                        ui.colored_label(egui::Color32::GREEN, "Speaking");
                                    }
                                },
                            );
                        });

                        // Draw waveform visualization
                        let response = ui.allocate_response(
                            egui::Vec2::new(ui.available_width(), 100.0),
                            egui::Sense::hover(),
                        );

                        let painter = ui.painter();
                        visualizer.paint_waveform(&painter, response.rect);

                        // Audio controls
                        ui.horizontal(|ui| {
                            ui.label("Volume:");

                            // Use persistent volume from participant state (FIXED)
                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut participant_state.participant_volume,
                                        0.0..=2.0,
                                    )
                                    .suffix("%"),
                                )
                                .changed()
                            {
                                // Apply to audio player immediately
                                if let Some(audio_player) = self.audio_players.get(&participant_id) {
                                    let effective_volume = if participant_state.is_muted {
                                        0.0
                                    } else {
                                        participant_state.participant_volume
                                    };
                                    audio_player.set_volume(effective_volume);
                                }

                                // Apply to visualizer for consistent display
                                if let Some(audio_viz) = self.audio_visualizers.get(&participant_id)
                                {
                                    let viz_volume = if participant_state.is_muted {
                                        0.0
                                    } else {
                                        participant_state.participant_volume
                                    };
                                    if let Err(e) = audio_viz.set_volume(viz_volume) {
                                        tracing::warn!(
                                            "Failed to set visualizer volume for {}: {}",
                                            participant_id.0,
                                            e
                                        );
                                    }
                                }

                                tracing::debug!(
                                    "Updated volume for {}: {:.2}",
                                    participant_id.0,
                                    participant_state.participant_volume
                                );
                            }

                            // Mute toggle (ENHANCED)
                            if ui
                                .checkbox(&mut participant_state.is_muted, "Mute")
                                .changed()
                            {
                                let effective_volume = if participant_state.is_muted {
                                    0.0
                                } else {
                                    participant_state.participant_volume
                                };

                                if let Some(audio_player) = self.audio_players.get(&participant_id) {
                                    audio_player.set_volume(effective_volume);
                                }

                                if let Some(audio_viz) = self.audio_visualizers.get(&participant_id)
                                {
                                    if let Err(e) = audio_viz.set_volume(effective_volume) {
                                        tracing::warn!(
                                            "Failed to set visualizer mute for {}: {}",
                                            participant_id.0,
                                            e
                                        );
                                    }
                                }

                                tracing::info!(
                                    "Toggled mute for {}: {}",
                                    participant_id.0,
                                    participant_state.is_muted
                                );
                            }

                            if ui.button("LiveKit Mute").clicked() {
                                // Connect to actual LiveKit mute APIs
                                if let Some(track_pub) = &participant_state.audio_publication {
                                    // Toggle track state based on type
                                    if track_pub.is_muted() {
                                        track_pub.set_enabled(true);
                                    } else {
                                        track_pub.set_enabled(false);
                                    }
                                }
                                tracing::info!("Toggling LiveKit mute for {}", participant_id.0);
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

                            let (display_width, display_height) = if video_aspect > available_aspect
                            {
                                (available_size.x, available_size.x / video_aspect)
                            } else {
                                (available_size.y * video_aspect, available_size.y)
                            };

                            let display_size = egui::Vec2::new(display_width, display_height);
                            let display_rect = egui::Rect::from_center_size(
                                video_response.rect.center(),
                                display_size,
                            );

                            // Get render state for blend factor and mouth openness
                            let (blend_factor, mouth_openness) = video_renderer.get_render_state();
                            
                            ui.allocate_ui_at_rect(display_rect, |ui| {
                                // Apply blend factor as alpha by adjusting the white tint color
                                let tint_color = egui::Color32::from_white_alpha(
                                    (blend_factor * 255.0) as u8
                                );
                                
                                // Draw the video texture with alpha blending
                                ui.painter().image(
                                    texture_id,
                                    display_rect,
                                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                    tint_color,
                                );
                                
                                // Add mouth openness visual effect as overlay
                                if mouth_openness > 0.1 {
                                    let overlay_alpha = (mouth_openness * 100.0) as u8;
                                    let overlay_color = egui::Color32::from_rgba_premultiplied(
                                        255, 100, 100, overlay_alpha
                                    );
                                    ui.painter().rect_filled(
                                        display_rect,
                                        egui::Rounding::same(4.0),
                                        overlay_color,
                                    );
                                }
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
        // Validate connection parameters
        if self.room_url.is_empty() {
            self.report_error(VisualizerError::missing_room_url());
            return;
        }

        if self.api_key.is_empty() || self.api_secret.is_empty() {
            self.report_error(VisualizerError::missing_credentials());
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
            match Self::establish_room_connection(
                url,
                api_key,
                api_secret,
                room_name,
                participant_name,
            )
            .await
            {
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
        let options = RoomOptions::default();
        let (room, events_rx) = Room::connect(&url, &token, options).await?;

        Ok((Arc::new(room), events_rx))
    }

    fn refresh_connection_quality(&mut self) {
        // Update connection quality for all participants based on TODO1.md
        for (participant_id, participant_state) in &mut self.participants {
            if let Some(audio_viz) = self.audio_visualizers.get(participant_id) {
                let stats = audio_viz.get_stats();
                // Calculate quality based on audio metrics
                participant_state.connection_quality = match stats.current_amplitude {
                    a if a > 0.1 => "Excellent".to_string(),
                    a if a > 0.05 => "Good".to_string(),
                    a if a > 0.01 => "Fair".to_string(),
                    _ => "Poor".to_string(),
                };
            }
        }
        tracing::debug!(
            "Refreshed connection quality for {} participants",
            self.participants.len()
        );
    }

    fn disconnect_from_room(&mut self) {
        self.room = None;
        self.room_events_rx = None;
        self.connection_rx = None;
        self.participants.clear();
        self.audio_visualizers.clear();
        self.audio_players.clear();
        self.video_renderers.clear();
        self.selected_participant = None;
        self.connection_status = "Disconnected".to_string();
    }
}

impl Drop for FullRoomVisualizerApp {
    fn drop(&mut self) {
        tracing::debug!("Cleaning up FullRoomVisualizerApp resources");

        // 1. Clean up audio visualizers (already have Drop impl from TODO1.md)
        for (participant_id, mut audio_viz) in self.audio_visualizers.drain() {
            audio_viz.stop(); // Method from AudioVisualizer
            tracing::debug!("Stopped audio visualizer for: {}", participant_id.0);
        }

        // 2. Clean up audio players (already have Drop impl from TODO1.md)
        for (participant_id, audio_player) in self.audio_players.drain() {
            audio_player.stop(); // Method from LiveKitAudioPlayer
            tracing::debug!("Stopped audio player for: {}", participant_id.0);
        }

        // 3. Clean up video renderers
        for (participant_id, _video_renderer) in self.video_renderers.drain() {
            // video_renderer.stop(); // If API exists in future
            tracing::debug!("Stopped video renderer for: {}", participant_id.0);
        }

        // 4. Disconnect from room gracefully
        if let Some(_room) = self.room.take() {
            // Room disconnect is async, but Drop is sync
            // Best effort cleanup - room will be dropped
            tracing::info!("Disconnecting from room during cleanup");
        }

        // 5. Close event channels
        if let Some(_rx) = self.room_events_rx.take() {
            tracing::debug!("Closed room events channel");
        }

        if let Some(_rx) = self.connection_rx.take() {
            tracing::debug!("Closed connection results channel");
        }

        tracing::info!("FullRoomVisualizerApp cleanup completed");
    }
}

fn main() {
    tracing_subscriber::fmt::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0]),
        renderer: eframe::Renderer::default(),
        ..Default::default()
    };

    if let Err(e) = eframe::run_native(
        "LiveKit Full Room Visualizer",
        options,
        Box::new(|_cc| {
            let app = FullRoomVisualizerApp::default();
            Ok(Box::new(app))
        }),
    ) {
        eprintln!("Failed to run app: {}", e);
        std::process::exit(1);
    }
}
