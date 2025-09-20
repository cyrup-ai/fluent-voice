use std::collections::VecDeque;

use crossterm::event::{Event, KeyCode};
use ratatui::{
    style::Style,
    text::Span,
    widgets::{Axis, GraphType},
};

use crate::audioio::Matrix;

use super::{DataSet, Dimension, DisplayMode, GraphConfig, UIMode, update_value_i};

use rustfft::{FftPlanner, num_complex::Complex};

/// Configuration for spectro display behavior
/// Follows the pattern established in src/visualizer_config.rs
const DEFAULT_SCALE_MULTIPLIER: f64 = 7.5;

/// Standard audio frequency markers for spectrum analysis
const STANDARD_AUDIO_FREQUENCIES: &[f64] = &[
    20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
    200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
    2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 20000.0
];

fn generate_frequency_markers(cfg: &GraphConfig, lower: f64, upper: f64) -> Vec<DataSet> {
    STANDARD_AUDIO_FREQUENCIES
        .iter()
        .map(|&freq| {
            let x = freq.ln();
            DataSet::new(
                None,
                vec![(x, lower), (x, upper)],
                cfg.marker_type,
                GraphType::Line,
                cfg.axis_color,
            )
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct SpectroDisplayConfig {
    /// Scale multiplier for Y-axis bounds calculation
    pub scale_multiplier: f64,
    /// Enable logarithmic scale display
    pub log_scale_enabled: bool,
    /// Custom axis label override
    pub custom_axis_label: Option<String>,
}

impl Default for SpectroDisplayConfig {
    fn default() -> Self {
        Self {
            scale_multiplier: DEFAULT_SCALE_MULTIPLIER,
            log_scale_enabled: false,
            custom_axis_label: None,
        }
    }
}

impl SpectroDisplayConfig {
    /// Create configuration with custom scale multiplier
    pub fn with_scale_multiplier(mut self, multiplier: f64) -> Self {
        self.scale_multiplier = multiplier.clamp(1.0, 20.0);
        self
    }

    /// Create configuration with logarithmic scale enabled
    pub fn with_log_scale(mut self, enabled: bool) -> Self {
        self.log_scale_enabled = enabled;
        self
    }

    /// Calculate axis configuration based on current settings
    pub fn calculate_axis_config(&self, cfg: &GraphConfig) -> (String, [f64; 2]) {
        let name = self.custom_axis_label
            .clone()
            .unwrap_or_else(|| {
                if self.log_scale_enabled { 
                    "| level".to_string() 
                } else { 
                    "| amplitude".to_string() 
                }
            });
        let bounds = [0.0, cfg.scale * self.scale_multiplier];
        (name, bounds)
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.scale_multiplier < 1.0 || self.scale_multiplier > 20.0 {
            return Err("Scale multiplier must be between 1.0 and 20.0");
        }
        Ok(())
    }
}

pub struct Spectrograph {
    pub sampling_rate: u32,
    pub buffer_size: u32,
    pub average: u32,
    pub buf: Vec<VecDeque<Vec<f64>>>,
    pub window: bool,
    pub log_y: bool,
    pub display_config: SpectroDisplayConfig,
}

fn magnitude(c: Complex<f64>) -> f64 {
    let squared = (c.re * c.re) + (c.im * c.im);
    squared.sqrt()
}

// got this from https://github.com/phip1611/spectrum-analyzer/blob/3c079ec2785b031d304bb381ff5f5fe04e6bcf71/src/windows.rs#L40
pub fn hann_window(samples: &[f64]) -> Vec<f64> {
    let mut windowed_samples = Vec::with_capacity(samples.len());
    let samples_len = samples.len() as f64;
    for (i, sample) in samples.iter().enumerate() {
        let two_pi_i = 2.0 * std::f64::consts::PI * i as f64;
        let idontknowthename = (two_pi_i / samples_len).cos();
        let multiplier = 0.5 * (1.0 - idontknowthename);
        windowed_samples.push(sample * multiplier)
    }
    windowed_samples
}

impl From<&crate::cfg::SourceOptions> for Spectrograph {
    fn from(value: &crate::cfg::SourceOptions) -> Self {
        Spectrograph {
            sampling_rate: value.sample_rate,
            buffer_size: value.buffer,
            average: 5,
            buf: Vec::new(),
            window: false,
            log_y: true,
            display_config: SpectroDisplayConfig::default(),
        }
    }
}

impl Spectrograph {
    /// Create with custom display configuration
    pub fn with_display_config(mut self, config: SpectroDisplayConfig) -> Self {
        self.display_config = config;
        self
    }
}

impl DisplayMode for Spectrograph {
    fn mode_str(&self) -> &'static str {
        "spectro"
    }

    fn channel_name(&self, index: usize) -> String {
        match index {
            0 => "L".into(),
            1 => "R".into(),
            _ => format!("{index}"),
        }
    }

    fn header(&self, _: &GraphConfig) -> String {
        let window_marker = if self.window { "-|-" } else { "---" };
        if self.average <= 1 {
            format!(
                "live  {}  {:.3}Hz bins",
                window_marker,
                self.sampling_rate as f64 / self.buffer_size as f64
            )
        } else {
            format!(
                "{}x avg ({:.1}s)  {}  {:.3}Hz bins",
                self.average,
                (self.average * self.buffer_size) as f64 / self.sampling_rate as f64,
                window_marker,
                self.sampling_rate as f64 / (self.buffer_size * self.average) as f64,
            )
        }
    }

    fn axis(&self, cfg: &GraphConfig, ui_mode: UIMode, dimension: Dimension) -> Axis {
        let (name, bounds) = match dimension {
            Dimension::X => (
                if self.log_x { "log(freq)" } else { "freq" },
                if self.log_x {
                    [20.0f64.ln(), 20000.0f64.ln()]
                } else {
                    [0.0, cfg.sampling_rate as f64 / 2.0]
                },
            ),
            Dimension::Y => {
                let (name, bounds) = self.display_config.calculate_axis_config(cfg);
                (name.as_str(), bounds)
            },
        };

        let mut a = Axis::default();
        if let UIMode::WithLabels = ui_mode {
            a = a.title(Span::styled(name, Style::default().fg(cfg.labels_color)));
        }
        a.style(Style::default().fg(cfg.axis_color)).bounds(bounds)
    }

    fn process(&mut self, cfg: &GraphConfig, data: &Matrix<f64>) -> Vec<DataSet> {
        if self.average == 0 {
            self.average = 1
        } // otherwise fft breaks
        if !cfg.pause {
            for (i, chan) in data.iter().enumerate() {
                if self.buf.len() <= i {
                    self.buf.push(VecDeque::new());
                }
                self.buf[i].push_back(chan.clone());
                while self.buf[i].len() > self.average as usize {
                    self.buf[i].pop_front();
                }
            }
        }

        let mut out = Vec::new();
        let mut planner: FftPlanner<f64> = FftPlanner::new();
        let sample_len = self.buffer_size * self.average;
        let resolution = self.sampling_rate as f64 / sample_len as f64;
        let fft = planner.plan_fft_forward(sample_len as usize);

        for (n, chan_queue) in self.buf.iter().enumerate().rev() {
            let mut chunk = chan_queue.iter().flatten().copied().collect::<Vec<f64>>();
            if self.window {
                chunk = hann_window(chunk.as_slice());
            }
            let mut max_val = chunk
                .iter()
                .max_by(|a, b| a.total_cmp(b))
                .copied()
                .unwrap_or(1.0);
            if max_val < 1. {
                max_val = 1.;
            }
            let mut tmp: Vec<Complex<f64>> = chunk
                .iter()
                .map(|x| Complex {
                    re: *x / max_val,
                    im: 0.0,
                })
                .collect();
            fft.process(tmp.as_mut_slice());
            out.push(DataSet::new(
                Some(self.channel_name(n)),
                tmp[..=tmp.len() / 2]
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        (
                            (i as f64 * resolution).ln(),
                            if self.log_y {
                                magnitude(*x).ln()
                            } else {
                                magnitude(*x)
                            },
                        )
                    })
                    .collect(),
                cfg.marker_type,
                if cfg.scatter {
                    GraphType::Scatter
                } else {
                    GraphType::Line
                },
                cfg.palette(n),
            ));
        }

        out
    }

    fn handle(&mut self, event: Event) {
        if let Event::Key(key) = event {
            match key.code {
                KeyCode::PageUp => update_value_i(&mut self.average, true, 1, 1., 1..65535),
                KeyCode::PageDown => update_value_i(&mut self.average, false, 1, 1., 1..65535),
                KeyCode::Char('w') => self.window = !self.window,
                KeyCode::Char('l') => self.log_y = !self.log_y,
                _ => {}
            }
        }
    }

    fn references(&self, cfg: &GraphConfig) -> Vec<DataSet> {
        let lower = 0.; // if self.log_y { -(cfg.scale * 5.) } else { 0. };
        let upper = cfg.scale * self.display_config.scale_multiplier;
        
        let mut markers = vec![
            // Base reference lines (horizontal)
            DataSet::new(
                None,
                vec![(0.0, 0.0), ((cfg.samples as f64).ln(), 0.0)],
                cfg.marker_type,
                GraphType::Line,
                cfg.axis_color,
            ),
        ];
        
        // Add generated frequency markers
        markers.extend(generate_frequency_markers(cfg, lower, upper));
        markers
    }
}
