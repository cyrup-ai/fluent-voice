use std::collections::BTreeMap;

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Widget},
};

/// Sent over the mpsc channel by xet_downloader.
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub path: String,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub speed_mbps: f64,
}

#[derive(Default)]
struct FileState {
    downloaded: u64,
    total: u64,
    speed: f64,
}

#[derive(Default)]
pub struct App {
    files: BTreeMap<String, FileState>,
}

impl App {
    pub fn update(&mut self, p: ProgressUpdate) {
        let e = self.files.entry(p.path).or_default();
        e.downloaded = p.bytes_downloaded;
        e.total = p.total_bytes.max(1); // avoid zero
        e.speed = p.speed_mbps;
    }

    pub fn draw(&self, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints(&[Constraint::Min(1), Constraint::Length(3)])
            .split(f.area());

        self.draw_list(f, chunks[0]);
        self.draw_footer(f, chunks[1]);
    }

    fn draw_list(&self, f: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self
            .files
            .iter()
            .map(|(name, state)| {
                let pct = state.downloaded as f64 / state.total as f64;
                let gauge = Gauge::default()
                    .ratio(pct)
                    .gauge_style(
                        Style::default()
                            .fg(Color::Cyan)
                            .bg(Color::DarkGray)
                            .add_modifier(Modifier::BOLD),
                    )
                    .label(Span::raw(format!(
                        "{:>6.1}% • {:>7.2} MiB/s",
                        pct * 100.0,
                        state.speed
                    )));

                // Render gauge to buffer line
                let mut buf = ratatui::buffer::Buffer::empty(Rect::new(0, 0, area.width, 1));
                Widget::render(gauge, area, &mut buf);
                let mut text = String::with_capacity(area.width as usize);
                for cell in buf.content() {
                    text.push(cell.symbol().chars().next().unwrap_or(' '));
                }

                ListItem::new(Line::from(vec![
                    Span::styled(name, Style::default().fg(Color::Yellow)),
                    Span::raw(" "),
                    Span::raw(text),
                ]))
            })
            .collect();

        let list =
            List::new(items).block(Block::default().title("Downloads").borders(Borders::ALL));
        f.render_widget(list, area);
    }

    fn draw_footer(&self, f: &mut Frame, area: Rect) {
        let (done, total): (u64, u64) = self
            .files
            .values()
            .fold((0, 0), |acc, s| (acc.0 + s.downloaded, acc.1 + s.total));

        let pct = if total == 0 {
            0.0
        } else {
            done as f64 / total as f64
        };

        let gauge = Gauge::default()
            .block(Block::default().title("Total").borders(Borders::ALL))
            .ratio(pct)
            .label(format!(
                "{:>6.1}% – {}/{} MiB",
                pct * 100.0,
                done / (1 << 20),
                total / (1 << 20)
            ));

        f.render_widget(gauge, area);
    }
}
