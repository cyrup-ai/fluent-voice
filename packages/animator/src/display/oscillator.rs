/// Oscillator display component
use crossterm::event::Event;
use ratatui::widgets::{Axis, GraphType};

use crate::{
    input::Matrix,
    oscillator::{DataSet, Dimension, DisplayMode, GraphConfig, UIMode},
};

/// Oscillator visualization component
#[derive(Debug, Clone, Default)]
pub struct Oscillator {
    /// Current channel being displayed
    pub channel: usize,
}

impl DisplayMode for Oscillator {
    #[allow(warnings)]
    fn axis(&self, cfg: &GraphConfig, ui_mode: UIMode, dimension: Dimension) -> Axis {
        match dimension {
            Dimension::X => {
                let mut axis = Axis::default()
                    .style(ratatui::style::Style::default().fg(cfg.axis_color))
                    .bounds([0.0, cfg.samples as f64]);
                if let UIMode::WithLabels = ui_mode {
                    axis = axis.title("Time");
                }
                axis
            }
            Dimension::Y => {
                let mut axis = Axis::default()
                    .style(ratatui::style::Style::default().fg(cfg.axis_color))
                    .bounds([-cfg.scale, cfg.scale]);
                if let UIMode::WithLabels = ui_mode {
                    axis = axis.title("Amplitude");
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
            .map(|(i, &value)| (i as f64, value))
            .collect();

        vec![DataSet::new(
            Some(format!("Channel {}", self.channel)),
            dataset_points,
            cfg.marker_type,
            GraphType::Line,
            cfg.palette(self.channel),
        )]
    }

    fn mode_str(&self) -> &'static str {
        "Oscillator"
    }

    fn channel_name(&self, index: usize) -> String {
        format!("Ch{index}")
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
                _ => {}
            }
        }
    }
}
