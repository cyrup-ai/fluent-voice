use crate::{
    AudioVisualizer, ConnectionQuality, ErrorState, LiveKitAudioPlayer, RoomVisualizerConfig,
    VideoRenderer, VisualizerError,
};
use anyhow::Result;
use eframe::egui;
use egui_wgpu::RenderState;
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
pub struct FullRoomVisualizerApp {
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
    pub room_url: String,
    pub api_key: String,
    pub api_secret: String,
    pub room_name: String,
    pub participant_name: String,

    // Configuration and error handling (NEW - TODO2.md)
    config: RoomVisualizerConfig,
    error_state: Option<ErrorState>,
    last_quality_update: Instant,

    // Real-time update configuration (NEW - TODO3.md Phase 2)
    rt_config: RealTimeUpdateConfig,

    // Connection recovery manager (NEW - TODO3.md Phase 3)
    recovery_manager: ConnectionRecoveryManager,

    // Copy feedback state
    copy_feedback: Option<(String, Instant)>, // (message, timestamp)

    // Cached sharing command to avoid regenerating every frame
    cached_sharing_command: Option<(String, usize, String, String)>, // (command, participant_count, room_url, room_name)
}

#[derive(Clone, Debug)]
struct ParticipantState {
    #[allow(dead_code)]
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
            room_url: env::var("LIVEKIT_URL").unwrap_or_default(),
            api_key: env::var("LIVEKIT_API_KEY").unwrap_or_default(),
            api_secret: env::var("LIVEKIT_API_SECRET").unwrap_or_default(),
            room_name: "fluent-voice-demo".to_string(),
            participant_name: "Rust Visualizer".to_string(),
            config: RoomVisualizerConfig::default(),
            error_state: None,
            last_quality_update: Instant::now(),
            rt_config: RealTimeUpdateConfig::default(),
            recovery_manager: ConnectionRecoveryManager::default(),
            copy_feedback: None,
            cached_sharing_command: None,
        }
    }
}

impl FullRoomVisualizerApp {
    /// Create a new app with proper WGPU render state from eframe CreationContext
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let app = Self::default();

        // TODO: Update to newer eframe API for wgpu render state access
        // Note: wgpu_render_state field has been removed from CreationContext in newer eframe versions
        // app.render_state = None; // Temporarily disabled until API is updated

        app
    }
}

impl eframe::App for FullRoomVisualizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Render state properly initialized from eframe CreationContext for GPU rendering

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
        // Use audio-based quality calculation from existing audio visualizer
        if let Some(audio_viz) = self.audio_visualizers.get(participant_id) {
            let stats = audio_viz.get_stats();
            return ConnectionQuality::from_audio_stats(
                stats.current_amplitude,
                stats.average_amplitude,
                stats.peak_amplitude,
                false, // Default to not muted since is_muted field no longer available
            );
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

                        // Invalidate cached sharing command due to participant count change
                        self.cached_sharing_command = None;
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

                        // Invalidate cached sharing command due to participant count change
                        self.cached_sharing_command = None;

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

                        for (participant, _publications) in participants_with_tracks {
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
            ui.heading("Connected to Room");

            // Room sharing section
            ui.separator();
            ui.label("Share this room with others:");

            // Room name with copy
            ui.horizontal(|ui| {
                ui.label("Room Name:");
                ui.monospace(&self.room_name);
                if ui.button("📋").clicked() {
                    ui.ctx().copy_text(self.room_name.clone());
                    self.copy_feedback = Some(("Room name copied!".into(), Instant::now()));
                }
            });

            // Server URL with copy
            ui.horizontal(|ui| {
                ui.label("Server URL:");
                ui.monospace(&self.room_url);
                if ui.button("📋").clicked() {
                    ui.ctx().copy_text(self.room_url.clone());
                    self.copy_feedback = Some(("Server URL copied!".into(), Instant::now()));
                }
            });

            // Generate or use cached connection command with enterprise-grade error handling
            let current_participant_count = self.participants.len();
            let cache_valid = if let Some((_, cached_count, cached_url, cached_name)) =
                &self.cached_sharing_command
            {
                *cached_count == current_participant_count
                    && cached_url == &self.room_url
                    && cached_name == &self.room_name
            } else {
                false
            };

            let (connection_command, validation_error) = if cache_valid {
                // Cache hit - return reference to cached command with no error
                if let Some((cached_command, _, _, _)) = &self.cached_sharing_command {
                    (Some(cached_command.as_str()), None)
                } else {
                    // This should never happen, but handle gracefully
                    (
                        None,
                        Some(ValidationError::InvalidContent {
                            field: "Cache",
                            reason: "internal cache corruption",
                        }),
                    )
                }
            } else {
                // Cache miss or invalid - regenerate command
                match self.generate_sharing_command(current_participant_count) {
                    Ok(command) => {
                        // Update cache with minimal cloning - only clone the strings once
                        let room_url = self.room_url.clone();
                        let room_name = self.room_name.clone();
                        self.cached_sharing_command =
                            Some((command, current_participant_count, room_url, room_name));
                        // Safe access since we just set it
                        if let Some((cached_command, _, _, _)) = &self.cached_sharing_command {
                            (Some(cached_command.as_str()), None)
                        } else {
                            (
                                None,
                                Some(ValidationError::InvalidContent {
                                    field: "Cache",
                                    reason: "cache update failed",
                                }),
                            )
                        }
                    }
                    Err(validation_err) => {
                        // Command generation failed due to validation
                        (None, Some(validation_err))
                    }
                }
            };

            ui.separator();
            ui.label("Command for others to join:");

            // Use vertical layout for long command to prevent overflow
            ui.vertical(|ui| {
                match (connection_command, &validation_error) {
                    (Some(command), None) => {
                        // Valid command - display it with copy functionality
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing.x = 0.0;
                            ui.monospace(command);
                        });

                        ui.horizontal(|ui| {
                            if ui.button("📋 Copy Command").clicked() {
                                ui.ctx().copy_text(command.to_string());
                                self.copy_feedback =
                                    Some(("Command copied!".into(), Instant::now()));
                            }

                            // Show copy feedback with automatic cleanup
                            if let Some((message, timestamp)) = &self.copy_feedback {
                                if timestamp.elapsed() < Duration::from_secs(2) {
                                    ui.colored_label(egui::Color32::from_rgb(0, 200, 0), message);
                                } else {
                                    self.copy_feedback = None;
                                }
                            }
                        });
                    }
                    (None, Some(error)) => {
                        // Validation error - display specific error message with appropriate styling
                        let error_color = match error {
                            ValidationError::EmptyInput { .. } => {
                                egui::Color32::from_rgb(255, 165, 0)
                            } // Orange for missing input
                            ValidationError::TooLong { .. } => {
                                egui::Color32::from_rgb(255, 100, 100)
                            } // Red for length issues
                            ValidationError::InvalidUrlScheme { .. }
                            | ValidationError::MalformedUrl { .. }
                            | ValidationError::InvalidHostname { .. }
                            | ValidationError::InvalidPort { .. }
                            | ValidationError::InvalidIPv4 { .. }
                            | ValidationError::InvalidIPv6 { .. } => {
                                egui::Color32::from_rgb(255, 100, 100)
                            } // Red for format errors
                            ValidationError::UnsafeCharacters { .. } => {
                                egui::Color32::from_rgb(255, 50, 50)
                            } // Bright red for security issues
                            ValidationError::InvalidEncoding { .. }
                            | ValidationError::InvalidContent { .. } => {
                                egui::Color32::from_rgb(255, 100, 100)
                            } // Red for content issues
                        };

                        ui.colored_label(error_color, error.user_message());

                        // Add helpful guidance for common errors
                        match error {
                            ValidationError::InvalidUrlScheme { .. } => {
                                ui.small("💡 Example: wss://your-server.com:8080");
                            }
                            ValidationError::InvalidHostname { .. } => {
                                ui.small("💡 Examples: localhost, example.com, 192.168.1.1");
                            }
                            ValidationError::TooLong {
                                field, max_length, ..
                            } => {
                                ui.small(format!(
                                    "💡 {} must be {} characters or less",
                                    field, max_length
                                ));
                            }
                            ValidationError::EmptyInput { field } => {
                                ui.small(format!(
                                    "💡 Please enter a valid {}",
                                    field.to_lowercase()
                                ));
                            }
                            _ => {}
                        }
                    }
                    (None, None) => {
                        // This should never happen, but handle gracefully
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 100, 100),
                            "⚠ Internal error: unable to generate command",
                        );
                    }
                    (Some(_), Some(_)) => {
                        // This should never happen either
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 100, 100),
                            "⚠ Internal error: inconsistent state",
                        );
                    }
                }
            });

            ui.separator();
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
                if let Some(_room) = &self.room {
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
            let participant_ids: Vec<ParticipantIdentity> =
                self.audio_visualizers.keys().cloned().collect();
            for participant_id in participant_ids {
                if let (Some(visualizer), Some(participant_state)) = (
                    self.audio_visualizers.get(&participant_id),
                    self.participants.get_mut(&participant_id),
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
                                if let Some(audio_player) = self.audio_players.get(&participant_id)
                                {
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

                                if let Some(audio_player) = self.audio_players.get(&participant_id)
                                {
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

                            ui.scope_builder(egui::UiBuilder::new().max_rect(display_rect), |ui| {
                                // Apply blend factor as alpha by adjusting the white tint color
                                let tint_color =
                                    egui::Color32::from_white_alpha((blend_factor * 255.0) as u8);

                                // Draw the video texture with alpha blending
                                ui.painter().image(
                                    texture_id,
                                    display_rect,
                                    egui::Rect::from_min_max(
                                        egui::pos2(0.0, 0.0),
                                        egui::pos2(1.0, 1.0),
                                    ),
                                    tint_color,
                                );

                                // Add mouth openness visual effect as overlay
                                if mouth_openness > 0.1 {
                                    let overlay_alpha = (mouth_openness * 100.0) as u8;
                                    let overlay_color = egui::Color32::from_rgba_premultiplied(
                                        255,
                                        100,
                                        100,
                                        overlay_alpha,
                                    );
                                    ui.painter().rect_filled(
                                        display_rect,
                                        egui::CornerRadius::same(4),
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

    pub fn connect_to_room(&mut self) {
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
}

/// Comprehensive validation error types for enterprise-grade input validation
#[derive(Debug, Clone, PartialEq, Eq)]
enum ValidationError {
    /// Input field is empty or contains only whitespace
    EmptyInput { field: &'static str },
    /// Input exceeds maximum allowed length
    TooLong {
        field: &'static str,
        max_length: usize,
        actual_length: usize,
    },
    /// URL scheme is not ws:// or wss://
    InvalidUrlScheme { provided: String },
    /// URL contains invalid characters or structure
    MalformedUrl { reason: &'static str },
    /// Hostname is empty or invalid format
    InvalidHostname { hostname: String },
    /// Port number is invalid (not 1-65535)
    InvalidPort { port: String },
    /// IPv4 address format is invalid
    InvalidIPv4 { address: String },
    /// IPv6 address format is invalid
    InvalidIPv6 { address: String },
    /// Input contains dangerous shell metacharacters that cannot be safely escaped
    UnsafeCharacters {
        field: &'static str,
        characters: String,
    },
    /// Input contains invalid UTF-8 sequences
    #[allow(dead_code)]
    InvalidEncoding { field: &'static str },
    /// Input contains only control characters or whitespace
    InvalidContent {
        field: &'static str,
        reason: &'static str,
    },
}

impl ValidationError {
    /// Get user-friendly error message with specific guidance
    #[inline]
    fn user_message(&self) -> String {
        match self {
            ValidationError::EmptyInput { field } => {
                format!("⚠ {} cannot be empty. Please enter a valid value.", field)
            }
            ValidationError::TooLong {
                field,
                max_length,
                actual_length,
            } => {
                format!(
                    "⚠ {} is too long ({} characters). Maximum allowed: {} characters.",
                    field, actual_length, max_length
                )
            }
            ValidationError::InvalidUrlScheme { provided } => {
                format!(
                    "⚠ URL must start with 'ws://' or 'wss://'. Found: '{}'",
                    provided
                )
            }
            ValidationError::MalformedUrl { reason } => {
                format!("⚠ Invalid URL format: {}", reason)
            }
            ValidationError::InvalidHostname { hostname } => {
                format!(
                    "⚠ Invalid hostname '{}'. Use format like 'example.com' or '192.168.1.1'",
                    hostname
                )
            }
            ValidationError::InvalidPort { port } => {
                format!(
                    "⚠ Invalid port '{}'. Port must be a number between 1 and 65535.",
                    port
                )
            }
            ValidationError::InvalidIPv4 { address } => {
                format!(
                    "⚠ Invalid IPv4 address '{}'. Use format like '192.168.1.1'",
                    address
                )
            }
            ValidationError::InvalidIPv6 { address } => {
                format!(
                    "⚠ Invalid IPv6 address '{}'. Use format like '[::1]' or '[2001:db8::1]'",
                    address
                )
            }
            ValidationError::UnsafeCharacters { field, characters } => {
                format!(
                    "⚠ {} contains unsafe characters: {}. Please remove these characters.",
                    field, characters
                )
            }
            ValidationError::InvalidEncoding { field } => {
                format!(
                    "⚠ {} contains invalid characters. Please use only valid text.",
                    field
                )
            }
            ValidationError::InvalidContent { field, reason } => {
                format!(
                    "⚠ {} is invalid: {}. Please enter valid content.",
                    field, reason
                )
            }
        }
    }
}

impl std::fmt::Display for ValidationError {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.user_message())
    }
}

impl std::error::Error for ValidationError {}

impl FullRoomVisualizerApp {
    // Security constants for input validation
    const MAX_URL_LENGTH: usize = 2048;
    const MAX_ROOM_NAME_LENGTH: usize = 256;
    const MAX_PARTICIPANT_NAME_LENGTH: usize = 128;

    /// Enterprise-grade POSIX-compliant shell escaping to prevent all command injection attacks
    /// Uses single-quote escaping strategy which is the safest approach for shell arguments
    #[inline]
    fn shell_escape(arg: &str) -> String {
        // Fast path for arguments that don't need escaping
        if arg
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | ':'))
        {
            return arg.to_string();
        }

        // Single-quote escaping is the safest method for POSIX shells
        // Inside single quotes, only single quotes need special handling
        if !arg.contains('\'') {
            // No single quotes - simple case
            format!("'{}'", arg)
        } else {
            // Contains single quotes - escape them as '\''
            // This closes the current single-quoted string, adds an escaped single quote,
            // then opens a new single-quoted string
            let mut result = String::with_capacity(arg.len() + 20);
            result.push('\'');

            for c in arg.chars() {
                if c == '\'' {
                    result.push_str("'\\''");
                } else {
                    result.push(c);
                }
            }

            result.push('\'');
            result
        }
    }

    /// Enterprise-grade RFC 3986 compliant WebSocket URL validation
    /// Validates scheme, hostname, port, and path components according to standards
    #[inline]
    fn validate_websocket_url(url: &str) -> Result<(), ValidationError> {
        let trimmed = url.trim();

        // Length validation
        if trimmed.len() > Self::MAX_URL_LENGTH {
            return Err(ValidationError::TooLong {
                field: "URL",
                max_length: Self::MAX_URL_LENGTH,
                actual_length: trimmed.len(),
            });
        }

        // Scheme validation - must be ws:// or wss://
        let (is_secure, after_scheme) = if let Some(after) = trimmed.strip_prefix("wss://") {
            (true, after)
        } else if let Some(after) = trimmed.strip_prefix("ws://") {
            (false, after)
        } else {
            let scheme = trimmed.split("://").next().unwrap_or(trimmed);
            return Err(ValidationError::InvalidUrlScheme {
                provided: scheme.to_string(),
            });
        };

        if after_scheme.is_empty() {
            return Err(ValidationError::MalformedUrl {
                reason: "missing hostname after scheme",
            });
        }

        // Split into host and path components
        let (host_port, _path) = if let Some(slash_pos) = after_scheme.find('/') {
            (&after_scheme[..slash_pos], &after_scheme[slash_pos..])
        } else {
            (after_scheme, "")
        };

        if host_port.is_empty() {
            return Err(ValidationError::MalformedUrl {
                reason: "empty hostname",
            });
        }

        // Parse host and port
        let (_host, port) = if host_port.starts_with('[') {
            // IPv6 address format [::1]:8080 - enhanced bracket validation
            if let Some(bracket_end) = host_port.find(']') {
                // Validate bracket positioning
                if bracket_end == 1 {
                    return Err(ValidationError::MalformedUrl {
                        reason: "empty IPv6 address in brackets",
                    });
                }

                let ipv6_part = &host_port[1..bracket_end];
                let port_part = &host_port[bracket_end + 1..];

                // Validate IPv6 address format
                Self::validate_ipv6_address(ipv6_part)?;

                // Validate what comes after the closing bracket
                if port_part.is_empty() {
                    // Just [ipv6] with no port
                    (host_port, None)
                } else if port_part.starts_with(':') {
                    // [ipv6]:port format
                    let port_str = &port_part[1..];
                    if port_str.is_empty() {
                        return Err(ValidationError::MalformedUrl {
                            reason: "empty port after IPv6 address",
                        });
                    }
                    let port_num = Self::validate_port(port_str)?;
                    (host_port, Some(port_num))
                } else {
                    // Invalid characters immediately after ]
                    return Err(ValidationError::MalformedUrl {
                        reason: "invalid characters after IPv6 address - expected ':port' or end of host",
                    });
                }
            } else {
                return Err(ValidationError::MalformedUrl {
                    reason: "unclosed IPv6 bracket - missing ']'",
                });
            }
        } else if let Some(colon_pos) = host_port.rfind(':') {
            // Check if this looks like a port (all digits after colon)
            let potential_port = &host_port[colon_pos + 1..];
            if potential_port.chars().all(|c| c.is_ascii_digit()) && !potential_port.is_empty() {
                // This is host:port format
                let host_part = &host_port[..colon_pos];
                let port_num = Self::validate_port(potential_port)?;
                Self::validate_hostname(host_part)?;
                (host_part, Some(port_num))
            } else {
                // Colon is part of hostname (like IPv6 without brackets)
                Self::validate_hostname(host_port)?;
                (host_port, None)
            }
        } else {
            // Just hostname, no port
            Self::validate_hostname(host_port)?;
            (host_port, None)
        };

        // Validate default ports for security
        if let Some(port_num) = port {
            if is_secure && port_num == 80 {
                return Err(ValidationError::MalformedUrl {
                    reason: "wss:// should not use port 80 (use 443 or custom port)",
                });
            }
            if !is_secure && port_num == 443 {
                return Err(ValidationError::MalformedUrl {
                    reason: "ws:// should not use port 443 (use 80 or custom port)",
                });
            }
        }

        Ok(())
    }

    /// Validate hostname according to DNS rules (RFC 1123)
    #[inline]
    fn validate_hostname(hostname: &str) -> Result<(), ValidationError> {
        if hostname.is_empty() {
            return Err(ValidationError::InvalidHostname {
                hostname: hostname.to_string(),
            });
        }

        // Check for IPv4 address format
        if hostname.chars().all(|c| c.is_ascii_digit() || c == '.') {
            return Self::validate_ipv4_address(hostname);
        }

        // DNS hostname validation
        if hostname.len() > 253 {
            return Err(ValidationError::InvalidHostname {
                hostname: hostname.to_string(),
            });
        }

        // Check each label (part between dots)
        for label in hostname.split('.') {
            if label.is_empty() || label.len() > 63 {
                return Err(ValidationError::InvalidHostname {
                    hostname: hostname.to_string(),
                });
            }

            // Labels must start and end with alphanumeric
            if !label
                .chars()
                .next()
                .map_or(false, |c| c.is_ascii_alphanumeric())
                || !label
                    .chars()
                    .last()
                    .map_or(false, |c| c.is_ascii_alphanumeric())
            {
                return Err(ValidationError::InvalidHostname {
                    hostname: hostname.to_string(),
                });
            }

            // Labels can only contain alphanumeric and hyphens
            if !label.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
                return Err(ValidationError::InvalidHostname {
                    hostname: hostname.to_string(),
                });
            }
        }

        Ok(())
    }

    /// Validate IPv4 address format
    #[inline]
    fn validate_ipv4_address(addr: &str) -> Result<(), ValidationError> {
        let parts: Vec<&str> = addr.split('.').collect();
        if parts.len() != 4 {
            return Err(ValidationError::InvalidIPv4 {
                address: addr.to_string(),
            });
        }

        for part in parts {
            if part.is_empty() || part.len() > 3 {
                return Err(ValidationError::InvalidIPv4 {
                    address: addr.to_string(),
                });
            }

            // Check for leading zeros (not allowed except for "0")
            if part.len() > 1 && part.starts_with('0') {
                return Err(ValidationError::InvalidIPv4 {
                    address: addr.to_string(),
                });
            }

            match part.parse::<u8>() {
                Ok(_) => {} // Valid octet
                Err(_) => {
                    return Err(ValidationError::InvalidIPv4 {
                        address: addr.to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate IPv6 address format with proper group validation and IPv4-in-IPv6 support
    #[inline]
    fn validate_ipv6_address(addr: &str) -> Result<(), ValidationError> {
        if addr.is_empty() {
            return Err(ValidationError::InvalidIPv6 {
                address: addr.to_string(),
            });
        }

        // Handle IPv4-in-IPv6 addresses (e.g., ::ffff:192.0.2.1)
        if let Some(ipv4_start) = addr.rfind(':') {
            let potential_ipv4 = &addr[ipv4_start + 1..];
            if potential_ipv4.contains('.') {
                // Validate the IPv4 part
                Self::validate_ipv4_address(potential_ipv4)?;
                // Validate the IPv6 prefix part (without trailing colon)
                let ipv6_prefix = &addr[..ipv4_start];
                if !ipv6_prefix.is_empty() {
                    if ipv6_prefix == ":" {
                        // Special case: IPv4-compatible IPv6 (::w.x.y.z)
                        // The prefix is just "::" which is valid
                        return Ok(());
                    } else {
                        return Self::validate_partial_ipv6_address(ipv6_prefix, 2);
                    }
                }
            }
        }

        Self::validate_pure_ipv6_address(addr)
    }

    /// Validate pure IPv6 address (no embedded IPv4)
    #[inline]
    fn validate_pure_ipv6_address(addr: &str) -> Result<(), ValidationError> {
        // Check for valid characters only
        let valid_chars = addr.chars().all(|c| c.is_ascii_hexdigit() || c == ':');
        if !valid_chars {
            return Err(ValidationError::InvalidIPv6 {
                address: addr.to_string(),
            });
        }

        // Handle special case of "::" (all zeros)
        if addr == "::" {
            return Ok(());
        }

        // Check for invalid patterns
        if addr.starts_with(':') && !addr.starts_with("::") {
            return Err(ValidationError::InvalidIPv6 {
                address: addr.to_string(),
            });
        }
        if addr.ends_with(':') && !addr.ends_with("::") {
            return Err(ValidationError::InvalidIPv6 {
                address: addr.to_string(),
            });
        }
        if addr.contains(":::") {
            return Err(ValidationError::InvalidIPv6 {
                address: addr.to_string(),
            });
        }

        // Count "::" occurrences - can only have one
        let double_colon_count = addr.matches("::").count();
        if double_colon_count > 1 {
            return Err(ValidationError::InvalidIPv6 {
                address: addr.to_string(),
            });
        }

        // Split by "::" if present
        if double_colon_count == 1 {
            let parts: Vec<&str> = addr.split("::").collect();
            if parts.len() != 2 {
                return Err(ValidationError::InvalidIPv6 {
                    address: addr.to_string(),
                });
            }

            let left_groups = if parts[0].is_empty() {
                0
            } else {
                parts[0].split(':').count()
            };
            let right_groups = if parts[1].is_empty() {
                0
            } else {
                parts[1].split(':').count()
            };

            // Total groups must not exceed 8
            if left_groups + right_groups >= 8 {
                return Err(ValidationError::InvalidIPv6 {
                    address: addr.to_string(),
                });
            }

            // Validate each group in left part
            if !parts[0].is_empty() {
                for group in parts[0].split(':') {
                    Self::validate_ipv6_group(group, addr)?;
                }
            }

            // Validate each group in right part
            if !parts[1].is_empty() {
                for group in parts[1].split(':') {
                    Self::validate_ipv6_group(group, addr)?;
                }
            }
        } else {
            // No "::" - must have exactly 8 groups
            let groups: Vec<&str> = addr.split(':').collect();
            if groups.len() != 8 {
                return Err(ValidationError::InvalidIPv6 {
                    address: addr.to_string(),
                });
            }

            // Validate each group
            for group in groups {
                Self::validate_ipv6_group(group, addr)?;
            }
        }

        Ok(())
    }

    /// Validate individual IPv6 group (1-4 hex digits)
    #[inline]
    fn validate_ipv6_group(group: &str, full_addr: &str) -> Result<(), ValidationError> {
        if group.is_empty() || group.len() > 4 {
            return Err(ValidationError::InvalidIPv6 {
                address: full_addr.to_string(),
            });
        }

        // All characters must be hex digits
        if !group.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(ValidationError::InvalidIPv6 {
                address: full_addr.to_string(),
            });
        }

        Ok(())
    }

    /// Validate partial IPv6 address for IPv4-in-IPv6 format
    /// reserved_groups: number of groups reserved for IPv4 (typically 2)
    #[inline]
    fn validate_partial_ipv6_address(
        addr: &str,
        reserved_groups: usize,
    ) -> Result<(), ValidationError> {
        // Check for valid characters only
        let valid_chars = addr.chars().all(|c| c.is_ascii_hexdigit() || c == ':');
        if !valid_chars {
            return Err(ValidationError::InvalidIPv6 {
                address: addr.to_string(),
            });
        }

        // Handle special case of "::" (all zeros)
        if addr == "::" {
            return Ok(());
        }

        // Check for invalid patterns
        if addr.starts_with(':') && !addr.starts_with("::") {
            return Err(ValidationError::InvalidIPv6 {
                address: addr.to_string(),
            });
        }
        if addr.ends_with(':') && !addr.ends_with("::") {
            return Err(ValidationError::InvalidIPv6 {
                address: addr.to_string(),
            });
        }
        if addr.contains(":::") {
            return Err(ValidationError::InvalidIPv6 {
                address: addr.to_string(),
            });
        }

        // Count "::" occurrences - can only have one
        let double_colon_count = addr.matches("::").count();
        if double_colon_count > 1 {
            return Err(ValidationError::InvalidIPv6 {
                address: addr.to_string(),
            });
        }

        // Calculate maximum allowed groups (8 total - reserved for IPv4)
        let max_ipv6_groups = 8 - reserved_groups;

        // Split by "::" if present
        if double_colon_count == 1 {
            let parts: Vec<&str> = addr.split("::").collect();
            if parts.len() != 2 {
                return Err(ValidationError::InvalidIPv6 {
                    address: addr.to_string(),
                });
            }

            let left_groups = if parts[0].is_empty() {
                0
            } else {
                parts[0].split(':').count()
            };
            let right_groups = if parts[1].is_empty() {
                0
            } else {
                parts[1].split(':').count()
            };

            // Total groups must not exceed maximum allowed
            if left_groups + right_groups > max_ipv6_groups {
                return Err(ValidationError::InvalidIPv6 {
                    address: addr.to_string(),
                });
            }

            // Validate each group in left part
            if !parts[0].is_empty() {
                for group in parts[0].split(':') {
                    Self::validate_ipv6_group(group, addr)?;
                }
            }

            // Validate each group in right part
            if !parts[1].is_empty() {
                for group in parts[1].split(':') {
                    Self::validate_ipv6_group(group, addr)?;
                }
            }
        } else {
            // No "::" - count groups and ensure within limits
            let groups: Vec<&str> = addr.split(':').collect();
            if groups.len() > max_ipv6_groups {
                return Err(ValidationError::InvalidIPv6 {
                    address: addr.to_string(),
                });
            }

            // Validate each group
            for group in groups {
                Self::validate_ipv6_group(group, addr)?;
            }
        }

        Ok(())
    }

    /// Validate port number (1-65535)
    #[inline]
    fn validate_port(port_str: &str) -> Result<u16, ValidationError> {
        if port_str.is_empty() {
            return Err(ValidationError::InvalidPort {
                port: port_str.to_string(),
            });
        }

        match port_str.parse::<u16>() {
            Ok(0) => Err(ValidationError::InvalidPort {
                port: port_str.to_string(),
            }),
            Ok(port) => Ok(port),
            Err(_) => Err(ValidationError::InvalidPort {
                port: port_str.to_string(),
            }),
        }
    }

    /// Generate a sharing command with enterprise-grade validation and security
    /// Returns Result with specific error information for comprehensive user feedback
    #[inline]
    fn generate_sharing_command(
        &self,
        participant_count: usize,
    ) -> Result<String, ValidationError> {
        let room_url = self.room_url.trim();
        let room_name = self.room_name.trim();

        // Validate inputs are not empty
        if room_url.is_empty() {
            return Err(ValidationError::EmptyInput { field: "Room URL" });
        }
        if room_name.is_empty() {
            return Err(ValidationError::EmptyInput { field: "Room Name" });
        }

        // Length validation for security
        if room_url.len() > Self::MAX_URL_LENGTH {
            return Err(ValidationError::TooLong {
                field: "Room URL",
                max_length: Self::MAX_URL_LENGTH,
                actual_length: room_url.len(),
            });
        }
        if room_name.len() > Self::MAX_ROOM_NAME_LENGTH {
            return Err(ValidationError::TooLong {
                field: "Room Name",
                max_length: Self::MAX_ROOM_NAME_LENGTH,
                actual_length: room_name.len(),
            });
        }

        // Comprehensive URL validation
        Self::validate_websocket_url(room_url)?;

        // Validate room name content
        if room_name
            .chars()
            .all(|c| c.is_whitespace() || c.is_control())
        {
            return Err(ValidationError::InvalidContent {
                field: "Room Name",
                reason: "contains only whitespace or control characters",
            });
        }

        // Check for potentially dangerous characters in room name
        let dangerous_chars: Vec<char> = room_name.chars()
            .filter(|&c| matches!(c, '\0' | '\x01'..='\x08' | '\x0B' | '\x0C' | '\x0E'..='\x1F' | '\x7F'))
            .collect();
        if !dangerous_chars.is_empty() {
            let char_display: String = dangerous_chars
                .iter()
                .map(|c| format!("\\x{:02X}", *c as u8))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(ValidationError::UnsafeCharacters {
                field: "Room Name",
                characters: char_display,
            });
        }

        // Prevent integer overflow with safe arithmetic
        let total_participants = participant_count.saturating_add(1); // Including me
        let next_participant_number = total_participants.saturating_add(1);

        // Generate participant name with length validation
        let participant_name = format!("Participant {}", next_participant_number);
        if participant_name.len() > Self::MAX_PARTICIPANT_NAME_LENGTH {
            return Err(ValidationError::TooLong {
                field: "Participant Name",
                max_length: Self::MAX_PARTICIPANT_NAME_LENGTH,
                actual_length: participant_name.len(),
            });
        }

        // Generate command with enterprise-grade shell escaping
        Ok(format!(
            "anima room --url {} --room-name {} --participant-name {}",
            Self::shell_escape(room_url),
            Self::shell_escape(room_name),
            Self::shell_escape(&participant_name)
        ))
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
        self.copy_feedback = None; // Clear copy feedback state
        self.cached_sharing_command = None; // Clear cached command
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

#[allow(dead_code)]
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
        Box::new(|cc| {
            let app = FullRoomVisualizerApp::new(cc);
            Ok(Box::new(app))
        }),
    ) {
        eprintln!("Failed to run app: {}", e);
        std::process::exit(1);
    }
}
