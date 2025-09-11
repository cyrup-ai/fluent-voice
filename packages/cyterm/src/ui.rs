//! Ratatui + Cosmic-Text split-pane UI.
//
//  ┌───────────────────────────────┐
//  │   Real-time Transcript        │
//  │   (dimmed ghost completions)  │
//  ├───────────────────────────────┤
//  │   Status  •  device / FPS     │
//  └───────────────────────────────┘

use std::time::Instant;

use ratatui::{
    Frame,
    backend::Backend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Paragraph},
};

const MAX_LINES: usize = 1_000;

/// What the draw routine needs from the rest of the app.
pub struct UiState {
    /// Finalized transcript.
    pub transcript: String,
    /// “Ghost” candidate completions (token, prob 0–1).
    pub ghosts: Vec<(String, f32)>,
    /// Whether ASR is currently inside a speech segment.
    pub recording: bool,
    /// Audio level 0–1 for a cute VU meter.
    pub level: f32,
    /// FPS tracker.
    last_frame: Instant,
    fps: f32,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            transcript: String::new(),
            ghosts: Vec::new(),
            recording: false,
            level: 0.0,
            last_frame: Instant::now(),
            fps: 0.0,
        }
    }
}

impl UiState {
    pub fn tick(&mut self) {
        let now = Instant::now();
        let dt = now - self.last_frame;
        self.fps = 1.0 / dt.as_secs_f32().max(1e-3);
        self.last_frame = now;
    }
}

/// Draw once.
pub fn draw<B: Backend>(f: &mut Frame, ui: &mut UiState) {
    ui.tick();

    // Split vertical: main area + status bar.
    let layout = Layout::new(
        Direction::Vertical,
        [Constraint::Min(2), Constraint::Length(1)],
    );
    let rects = layout.split(f.area());
    let body = rects[0];
    let status = rects[1];

    // == transcript =========================================================
    let mut lines = Text::from(ui.transcript.clone()).lines;
    while lines.len() > MAX_LINES {
        lines.remove(0);
    }

    // Append dimmed ghost completions.
    for (token, p) in &ui.ghosts {
        lines.push(Line::from(Span::styled(
            token,
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )));
        // faint probability tooltip
        lines.push(Line::from(Span::styled(
            format!("{p:.2} "),
            Style::default().fg(Color::Gray),
        )));
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::new()
                .title("Transcript")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .wrap(ratatui::widgets::Wrap { trim: true });
    f.render_widget(paragraph, body);

    // == status bar =========================================================
    let status_text = Line::from(vec![
        Span::styled(
            if ui.recording { "● REC " } else { "    " },
            Style::default().fg(if ui.recording {
                Color::Red
            } else {
                Color::DarkGray
            }),
        ),
        Span::raw(format!(
            "│ mic: {:03}% │ FPS: {:.0}",
            (ui.level * 100.0) as u8,
            ui.fps
        )),
    ]);
    let status_paragraph = Paragraph::new(status_text).block(
        Block::new()
            .borders(Borders::TOP)
            .border_type(BorderType::Plain),
    );
    f.render_widget(status_paragraph, status);
}
