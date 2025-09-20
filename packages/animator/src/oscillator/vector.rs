use ratatui::{
    style::Style,
    text::Span,
    widgets::{Axis, GraphType},
};

use crate::input::Matrix;

use super::{DataSet, Dimension, DisplayMode, GraphConfig, UIMode};

/// Configuration for vector display data splitting
/// Based on patterns from src/visualizer_config.rs
#[derive(Debug, Clone)]
pub struct VectorDisplayConfig {
    /// Maximum points per dataset before splitting
    pub max_points_per_dataset: usize,
    /// Enable automatic data splitting
    pub auto_split_enabled: bool,
    /// Number of split parts (2 = split in half, 3 = split in thirds, etc.)
    pub split_parts: usize,
    /// Overlap between split parts (0.0 = no overlap, 0.1 = 10% overlap)
    pub split_overlap: f64,
}

impl Default for VectorDisplayConfig {
    fn default() -> Self {
        Self {
            max_points_per_dataset: 1000,
            auto_split_enabled: true,
            split_parts: 2, // Current hardcoded behavior: split in half
            split_overlap: 0.0,
        }
    }
}

impl VectorDisplayConfig {
    /// Create configuration with custom split parts
    pub fn with_split_parts(mut self, parts: usize) -> Self {
        self.split_parts = parts.clamp(1, 10);
        self
    }

    /// Create configuration with custom max points
    pub fn with_max_points(mut self, max_points: usize) -> Self {
        self.max_points_per_dataset = max_points.clamp(100, 10000);
        self
    }

    /// Split data according to configuration
    pub fn split_data_if_needed(&self, data: Vec<(f64, f64)>) -> Vec<Vec<(f64, f64)>> {
        if !self.auto_split_enabled || data.len() <= self.max_points_per_dataset {
            return vec![data];
        }
        
        if self.split_parts <= 1 {
            return vec![data];
        }

        let chunk_size = data.len() / self.split_parts;
        let overlap_size = (chunk_size as f64 * self.split_overlap) as usize;
        
        (0..self.split_parts)
            .map(|i| {
                let start = (i * chunk_size).saturating_sub(overlap_size);
                let end = ((i + 1) * chunk_size + overlap_size).min(data.len());
                data[start..end].to_vec()
            })
            .collect()
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.split_parts == 0 || self.split_parts > 10 {
            return Err("Split parts must be between 1 and 10");
        }
        if self.split_overlap < 0.0 || self.split_overlap > 0.5 {
            return Err("Split overlap must be between 0.0 and 0.5");
        }
        if self.max_points_per_dataset < 100 || self.max_points_per_dataset > 10000 {
            return Err("Max points per dataset must be between 100 and 10000");
        }
        Ok(())
    }
}

pub struct Vector {
    pub display_config: VectorDisplayConfig,
}

impl Default for Vector {
    fn default() -> Self {
        Self {
            display_config: VectorDisplayConfig::default(),
        }
    }
}

impl Vector {
    /// Create with custom display configuration
    pub fn with_display_config(config: VectorDisplayConfig) -> Self {
        Self {
            display_config: config,
        }
    }
}

impl DisplayMode for Vector {
    fn mode_str(&self) -> &'static str {
        "vector"
    }

    fn channel_name(&self, index: usize) -> String {
        format!("{index}")
    }

    fn header(&self, _: &GraphConfig) -> String {
        "live".into()
    }

    fn axis(&self, cfg: &GraphConfig, ui_mode: UIMode, dimension: Dimension) -> Axis {
        let (name, bounds) = match dimension {
            Dimension::X => ("left/right", [-cfg.scale, cfg.scale]),
            Dimension::Y => ("up/down", [-cfg.scale, cfg.scale]),
        };

        let mut a = Axis::default();
        if let UIMode::WithLabels = ui_mode {
            a = a.title(Span::styled(name, Style::default().fg(cfg.labels_color)));
        }
        a.style(Style::default().fg(cfg.axis_color)).bounds(bounds)
    }

    fn references(&self, cfg: &GraphConfig) -> Vec<DataSet> {
        vec![
            DataSet::new(
                None,
                vec![(-cfg.scale, 0.0), (cfg.scale, 0.0)],
                cfg.marker_type,
                GraphType::Line,
                cfg.axis_color,
            ),
            DataSet::new(
                None,
                vec![(0.0, -cfg.scale), (0.0, cfg.scale)],
                cfg.marker_type,
                GraphType::Line,
                cfg.axis_color,
            ),
        ]
    }

    fn process(&mut self, cfg: &GraphConfig, data: &Matrix<f64>) -> Vec<DataSet> {
        let mut out = Vec::new();

        for (n, chunk) in data.chunks(2).enumerate() {
            let mut tmp = vec![];
            match chunk.len() {
                2 => {
                    for i in 0..std::cmp::min(chunk[0].len(), chunk[1].len()) {
                        if i > cfg.samples as usize {
                            break;
                        }
                        tmp.push((chunk[0][i], chunk[1][i]));
                    }
                }
                1 => {
                    for i in 0..chunk[0].len() {
                        if i > cfg.samples as usize {
                            break;
                        }
                        tmp.push((chunk[0][i], i as f64));
                    }
                }
                _ => continue,
            }
            // Split data according to configuration
            let split_datasets = self.display_config.split_data_if_needed(tmp);
            for (split_idx, split_data) in split_datasets.into_iter().enumerate() {
                out.push(DataSet::new(
                    Some(self.channel_name((n * self.display_config.split_parts) + split_idx)),
                    split_data,
                cfg.marker_type,
                if cfg.scatter {
                    GraphType::Scatter
                } else {
                    GraphType::Line
                },
                    cfg.palette((n * self.display_config.split_parts) + split_idx),
                ));
            }
        }

        out
    }
}
