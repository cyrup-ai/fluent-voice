use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::{
    Terminal,
    backend::Backend,
    layout::{Constraint, Rect},
    style::Color,
    style::{Modifier, Style},
    symbols::Marker,
    widgets::Chart,
    widgets::{Cell, Row, Table},
};
use std::{
    io,
    time::{Duration, Instant},
};

use crate::{
    display::{oscillator::Oscillator, spectroscope::Spectrograph},
    input::{DataSource, Matrix},
    oscillator::{
        Dimension, DisplayMode, GraphConfig, update_value_f, update_value_i, vector::Vector,
    },
};

pub enum CurrentDisplayMode {
    Oscillator,
    Vector,
    Spectrograph,
}

pub struct App {
    #[allow(unused)]
    channels: u8,
    graph: GraphConfig,
    oscillator: Oscillator,
    vector: Vector,
    spectrograph: Spectrograph,
    mode: CurrentDisplayMode,
}

// Refactored: App::new now takes owned config objects, not CLI args directly.
impl App {
    pub fn new(ui: &crate::cfg::UiOptions, source: &crate::cfg::SourceOptions) -> Self {
        let graph = GraphConfig {
            axis_color: Color::DarkGray,
            labels_color: Color::Cyan,
            palette: vec![Color::Red, Color::Yellow, Color::Green, Color::Magenta],
            scale: ui.scale as f64,
            width: source.buffer,
            samples: source.buffer,
            sampling_rate: source.sample_rate,
            references: !ui.no_reference,
            show_ui: !ui.no_ui,
            scatter: ui.scatter,
            pause: false,
            marker_type: if ui.no_braille {
                Marker::Dot
            } else {
                Marker::Braille
            },
        };

        let oscillator = Oscillator::default();
        let vector = Vector::default();
        let spectrograph = Spectrograph::default();

        App {
            graph,
            oscillator,
            vector,
            spectrograph,
            mode: CurrentDisplayMode::Oscillator,
            channels: source.channels as u8,
        }
    }

    pub fn run<T: Backend>(
        &mut self,
        mut source: Box<dyn DataSource<f64>>,
        terminal: &mut Terminal<T>,
    ) -> Result<()> {
        let mut fps = 0;
        let mut framerate = 0;
        let mut last_poll = Instant::now();
        let mut channels = Matrix::default();

        loop {
            let data = source.recv().ok_or(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "data source returned null",
            ))?;

            if !self.graph.pause {
                channels = data;
            }

            fps += 1;

            if last_poll.elapsed().as_secs() >= 1 {
                framerate = fps;
                fps = 0;
                last_poll = Instant::now();
            }

            {
                let mut datasets = Vec::new();
                let graph = self.graph.clone(); // This is necessary to avoid borrowing issues.
                if self.graph.references {
                    datasets.append(&mut self.current_display_mut().references(&graph));
                }
                datasets.append(&mut self.current_display_mut().process(&graph, &channels));
                if let Err(e) = terminal.draw(|f| {
                    let mut size = f.area();
                    if self.graph.show_ui {
                        f.render_widget(
                            make_header(
                                &self.graph,
                                &self.current_display().header(&self.graph),
                                self.current_display().mode_str(),
                                framerate,
                                self.graph.pause,
                            ),
                            Rect {
                                x: size.x,
                                y: size.y,
                                width: size.width,
                                height: 1,
                            }, // a 1px line at the top
                        );
                        size.height -= 1;
                        size.y += 1;
                    }
                    let chart = Chart::new(datasets.iter().map(|x| x.into()).collect())
                        .x_axis(self.current_display().axis(&self.graph, Dimension::X))
                        .y_axis(self.current_display().axis(&self.graph, Dimension::Y));
                    f.render_widget(chart, size)
                }) {
                    return Err(anyhow::anyhow!("Terminal draw error: {}", e));
                }
            }

            while event::poll(Duration::from_millis(0))? {
                // process all enqueued events
                let event = event::read()?;

                if self.process_events(event.clone())? {
                    return Ok(());
                }
                self.current_display_mut().handle(event);
            }
        }
    }

    fn current_display_mut(&mut self) -> &mut dyn DisplayMode {
        match self.mode {
            CurrentDisplayMode::Oscillator => &mut self.oscillator as &mut dyn DisplayMode,
            CurrentDisplayMode::Vector => &mut self.vector as &mut dyn DisplayMode,
            CurrentDisplayMode::Spectrograph => &mut self.spectrograph as &mut dyn DisplayMode,
        }
    }

    fn current_display(&self) -> &dyn DisplayMode {
        match self.mode {
            CurrentDisplayMode::Oscillator => &self.oscillator as &dyn DisplayMode,
            CurrentDisplayMode::Vector => &self.vector as &dyn DisplayMode,
            CurrentDisplayMode::Spectrograph => &self.spectrograph as &dyn DisplayMode,
        }
    }

    fn process_events(&mut self, event: Event) -> Result<bool> {
        let mut quit = false;
        if let Event::Key(key) = event {
            if let KeyModifiers::CONTROL = key.modifiers {
                match key.code {
                    // mimic other programs shortcuts to quit, for user friendliness
                    KeyCode::Char('c') | KeyCode::Char('q') | KeyCode::Char('w') => quit = true,
                    _ => {}
                }
            }
            let magnitude = match key.modifiers {
                KeyModifiers::SHIFT => 10.0,
                KeyModifiers::CONTROL => 5.0,
                KeyModifiers::ALT => 0.2,
                _ => 1.0,
            };
            match key.code {
                KeyCode::Up => update_value_f(&mut self.graph.scale, 0.01, magnitude, 0.0..10.0), // inverted to act as zoom
                KeyCode::Down => update_value_f(&mut self.graph.scale, -0.01, magnitude, 0.0..10.0), // inverted to act as zoom
                KeyCode::Right => update_value_i(
                    &mut self.graph.samples,
                    true,
                    25,
                    magnitude,
                    0..self.graph.width * 2,
                ),
                KeyCode::Left => update_value_i(
                    &mut self.graph.samples,
                    false,
                    25,
                    magnitude,
                    0..self.graph.width * 2,
                ),
                KeyCode::Char('q') => quit = true,
                KeyCode::Char(' ') => self.graph.pause = !self.graph.pause,
                KeyCode::Char('s') => self.graph.scatter = !self.graph.scatter,
                KeyCode::Char('h') => self.graph.show_ui = !self.graph.show_ui,
                KeyCode::Char('r') => self.graph.references = !self.graph.references,
                KeyCode::Tab => {
                    // switch modes
                    match self.mode {
                        CurrentDisplayMode::Oscillator => self.mode = CurrentDisplayMode::Vector,
                        CurrentDisplayMode::Vector => self.mode = CurrentDisplayMode::Spectrograph,
                        CurrentDisplayMode::Spectrograph => {
                            self.mode = CurrentDisplayMode::Oscillator
                        }
                    }
                }
                KeyCode::Esc => {
                    self.graph.samples = self.graph.width;
                    self.graph.scale = 1.;
                }
                _ => {}
            }
        };

        Ok(quit)
    }
}

// The make_header function is used for rendering the UI header and is best kept here for clarity.

fn make_header<'a>(
    cfg: &GraphConfig,
    module_header: &'a str,
    kind_o_scope: &'static str,
    fps: usize,
    pause: bool,
) -> Table<'a> {
    Table::new(
        vec![Row::new(vec![
            Cell::from(format!("{kind_o_scope}::scope-tui")).style(
                Style::default()
                    .fg(*cfg.palette.first().expect("empty palette?"))
                    .add_modifier(Modifier::BOLD),
            ),
            Cell::from(module_header),
            Cell::from(format!("-{:.2}x+", cfg.scale)),
            Cell::from(format!("{}/{} spf", cfg.samples, cfg.width)),
            Cell::from(format!("{fps}fps")),
            Cell::from(if cfg.scatter { "***" } else { "---" }),
            Cell::from(if pause { "||" } else { "|>" }),
        ])],
        vec![
            Constraint::Percentage(35),
            Constraint::Percentage(25),
            Constraint::Percentage(7),
            Constraint::Percentage(13),
            Constraint::Percentage(6),
            Constraint::Percentage(6),
            Constraint::Percentage(6),
        ],
    )
    .style(Style::default().fg(cfg.labels_color))
}
