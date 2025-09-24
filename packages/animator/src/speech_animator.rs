use egui::load::SizedTexture;
use egui_wgpu::RenderState;
use livekit::webrtc::prelude::*;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::oneshot;

use crate::audio_visualizer::AudioVisualizer;
use crate::video_renderer::VideoRenderer;

/// Animates speech by synchronizing video frames with audio input and phoneme data
pub struct SpeechAnimator {
    audio_visualizer: AudioVisualizer,
    video_renderer: VideoRenderer,
    lip_sync_state: Arc<Mutex<LipSyncState>>,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

/// Represents the current state of lip synchronization
#[derive(Debug, Clone)]
struct LipSyncState {
    #[allow(dead_code)]
    current_phoneme: String,
    mouth_openness: f32,
    transition_progress: f32,
    audio_amplitude: f32,
}

impl Default for LipSyncState {
    fn default() -> Self {
        Self {
            current_phoneme: String::new(),
            mouth_openness: 0.0,
            transition_progress: 0.0,
            audio_amplitude: 0.0,
        }
    }
}

impl SpeechAnimator {
    /// Creates a new SpeechAnimator with the given audio and video tracks
    pub fn new(
        rt_handle: &tokio::runtime::Handle,
        render_state: RenderState,
        audio_track: RtcAudioTrack,
        video_track: RtcVideoTrack,
    ) -> Self {
        let audio_visualizer = AudioVisualizer::new(rt_handle, audio_track);
        let video_renderer = VideoRenderer::new(rt_handle, render_state, video_track);
        let lip_sync_state = Arc::new(Mutex::new(LipSyncState::default()));

        let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
        let lip_sync_state_clone = lip_sync_state.clone();
        let audio_visualizer_clone = audio_visualizer.clone();

        rt_handle.spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(50));

            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => {
                        break;
                    }
                    _ = interval.tick() => {
                        // Update lip sync state based on audio analysis
                        let amplitudes = audio_visualizer_clone.get_amplitudes();
                        let avg_amplitude = amplitudes.iter().sum::<f32>() / amplitudes.len().max(1) as f32;

                        let mut state = lip_sync_state_clone.lock();
                        state.audio_amplitude = avg_amplitude;

                        // Update mouth openness based on audio amplitude
                        let target_openness = (avg_amplitude * 2.0).min(1.0);
                        state.mouth_openness = state.mouth_openness * 0.7 + target_openness * 0.3;

                        // Smoothly transition between states
                        state.transition_progress = (state.transition_progress + 0.1).min(1.0);
                    }
                }
            }
        });

        Self {
            audio_visualizer,
            video_renderer,
            lip_sync_state,
            shutdown_tx: Some(shutdown_tx),
        }
    }

    /// Updates the animation state based on current audio and phoneme data
    pub fn update_animation(&self) {
        let state = self.lip_sync_state.lock();

        // Apply the current lip sync state to the video renderer
        if state.transition_progress < 1.0 {
            // Interpolate between states during transitions
            let progress = ease_in_out_cubic(state.transition_progress);
            self.video_renderer.set_blend_factor(progress);
        }

        // Update any video effects based on mouth openness
        self.video_renderer.set_mouth_openness(state.mouth_openness);
    }

    /// Renders the animated video and audio visualization
    pub fn render(&self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            // Render the animated video frame
            if let Some(texture_id) = self.video_renderer.texture_id() {
                let size = self.video_renderer.resolution();
                let aspect_ratio = size.0 as f32 / size.1 as f32;
                let available_width = ui.available_width();
                let display_size = egui::Vec2::new(available_width, available_width / aspect_ratio);

                ui.image(SizedTexture::new(texture_id, display_size));
            }

            // Render audio visualization
            ui.add_space(10.0);
            self.render_audio_waveform(ui);
        });
    }

    /// Renders the audio waveform visualization
    fn render_audio_waveform(&self, ui: &mut egui::Ui) {
        let amplitudes = self.audio_visualizer.get_amplitudes();
        if amplitudes.is_empty() {
            return;
        }

        let available_rect = ui.available_rect_before_wrap();
        let height = 50.0;
        let rect = egui::Rect::from_min_size(
            available_rect.min,
            egui::Vec2::new(available_rect.width(), height),
        );

        ui.allocate_rect(rect, egui::Sense::hover());

        let painter = ui.painter();
        let stroke = egui::Stroke::new(2.0, ui.visuals().text_color());

        let points: Vec<egui::Pos2> = amplitudes
            .iter()
            .enumerate()
            .map(|(i, &amp)| {
                let x = rect.left() + (i as f32 / amplitudes.len() as f32) * rect.width();
                let y = rect.center().y - amp * height * 0.5;
                egui::Pos2::new(x, y)
            })
            .collect();

        if points.len() > 1 {
            painter.add(egui::Shape::line(points, stroke));
        }
    }
}

impl Drop for SpeechAnimator {
    fn drop(&mut self) {
        // Signal the background task to shutdown
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

/// Cubic ease-in-out function for smooth transitions
fn ease_in_out_cubic(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    }
}
