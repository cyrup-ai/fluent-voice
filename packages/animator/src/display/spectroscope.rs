/// Spectroscope display component for frequency domain visualization
use crossterm::event::Event;
use ratatui::widgets::{Axis, GraphType};

use crate::{
    input::Matrix,
    oscillator::{DataSet, Dimension, DisplayMode, GraphConfig, UIMode},
};

/// Spectroscope visualization component for frequency analysis
#[derive(Debug, Clone)]
pub struct Spectrograph {
    /// Current channel being displayed
    pub channel: usize,
    /// FFT window size
    pub window_size: usize,
}

impl Default for Spectrograph {
    fn default() -> Self {
        Self {
            channel: 0,
            window_size: 1024,
        }
    }
}

impl DisplayMode for Spectrograph {
    #[allow(warnings)]
    fn axis(&self, cfg: &GraphConfig, ui_mode: UIMode, dimension: Dimension) -> Axis {
        match dimension {
            Dimension::X => {
                let mut axis = Axis::default()
                    .style(ratatui::style::Style::default().fg(cfg.axis_color))
                    .bounds([0.0, (cfg.sampling_rate / 2) as f64]);
                if let UIMode::WithLabels = ui_mode {
                    axis = axis.title("Frequency (Hz)");
                }
                axis
            }
            Dimension::Y => {
                let mut axis = Axis::default()
                    .style(ratatui::style::Style::default().fg(cfg.axis_color))
                    .bounds([-120.0, 0.0]);
                if let UIMode::WithLabels = ui_mode {
                    axis = axis.title("Magnitude (dB)");
                }
                axis
            }
        }
    }

    fn process(&mut self, cfg: &GraphConfig, data: &Matrix<f64>) -> Vec<DataSet> {
        if data.is_empty() || self.channel >= data.len() {
            return vec![];
        }

        let channel_data = &data[self.channel];
        let dataset_points: Vec<(f64, f64)> = channel_data
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                let freq = (i as f64 * cfg.sampling_rate as f64) / self.window_size as f64;
                let magnitude_db = if value > 0.0 {
                    20.0 * value.abs().log10()
                } else {
                    -120.0
                };
                (freq, magnitude_db)
            })
            .collect();

        vec![DataSet::new(
            Some(format!("Spectrum Ch{}", self.channel)),
            dataset_points,
            cfg.marker_type,
            GraphType::Line,
            cfg.palette(self.channel),
        )]
    }

    fn mode_str(&self) -> &'static str {
        "Spectrograph"
    }

    fn channel_name(&self, index: usize) -> String {
        format!("Spec{index}")
    }

    fn handle(&mut self, event: Event) {
        if let Event::Key(key) = event {
            match key.code {
                crossterm::event::KeyCode::Up => {
                    if self.channel > 0 {
                        self.channel -= 1;
                    }
                }
                crossterm::event::KeyCode::Down => {
                    self.channel += 1;
                }
                crossterm::event::KeyCode::Left => {
                    if self.window_size > 256 {
                        self.window_size /= 2;
                    }
                }
                crossterm::event::KeyCode::Right => {
                    if self.window_size < 8192 {
                        self.window_size *= 2;
                    }
                }
                _ => {}
            }
        }
    }
}
