use std::{collections::HashSet, io};

use cosmic_text::{
    Attrs, AttrsList, Buffer as CosmicBuffer, CacheKeyFlags, Family, FontSystem, LineEnding,
    Metrics, SwashCache, Weight, Wrap, fontdb::Database,
};
use ratatui::{
    backend::{Backend, ClearType, WindowSize},
    buffer::{Buffer, Cell},
    layout::{Position, Rect, Size},
    style::Modifier,
};

use crate::{
    colors::{blend_rgba, dim_rgb, rat_to_rgb},
    pixmap::RgbPixmap,
};

/// Pure-software ratatui backend that renders into an internal RGB frame-buffer.
pub struct SoftBackend {
    /// Ratatui logical buffer.
    buffer: Buffer,
    /// Software back-buffer.
    rgb: RgbPixmap,
    /// Font/shape state.
    font_sys: FontSystem,
    cosmic_buf: CosmicBuffer,
    cache: SwashCache,
    cw: usize,
    ch: usize,
    /// Blink bookkeeping.
    blink_tick: u16,
    blink_fast: bool,
    blink_slow: bool,
    /// Always-redraw list for blink/cursor cells.
    dirty: HashSet<(u16, u16)>,
    /// Cursor state.
    cursor_visible: bool,
    cur_pos: (u16, u16),
}

impl SoftBackend {
    /* ---------- public convenience API ---------- */

    /// Return raw RGB byte-slice (for screenshot / FFI).
    #[inline]
    pub fn rgb(&self) -> &[u8] {
        self.rgb.data()
    }
    /// Width in *pixels*.
    #[inline]
    pub fn pw(&self) -> usize {
        self.rgb.width()
    }
    /// Height in *pixels*.
    #[inline]
    pub fn ph(&self) -> usize {
        self.rgb.height()
    }

    /// Adjust font-size; recreates back-buffer & triggers full redraw.
    pub fn set_font_size(&mut self, px: i32) {
        let m = Metrics::new(px as f32, px as f32);
        self.cosmic_buf.set_metrics(&mut self.font_sys, m);
        self.recalc_cell_metrics();
        self.rgb = RgbPixmap::new(
            self.cw * self.buffer.area.width as usize,
            self.ch * self.buffer.area.height as usize,
        );
        self.redraw_all();
    }

    /// Build with a specific TrueType/OTF blob.
    pub fn new_with_font(cols: u16, rows: u16, px: i32, font_bytes: &[u8]) -> Self {
        let mut db = Database::new();
        db.load_font_data(font_bytes.to_vec());
        let fs = FontSystem::new_with_locale_and_db("en".into(), db);
        Self::init(cols, rows, px, fs)
    }

    /// Build using system font fallback (host font-config / CoreText / DirectWrite).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_system(cols: u16, rows: u16, px: i32) -> Self {
        Self::init(cols, rows, px, FontSystem::new())
    }

    /* ---------- internal helpers ---------- */

    fn init(cols: u16, rows: u16, px: i32, mut font_sys: FontSystem) -> Self {
        let metrics = Metrics::new(px as f32, px as f32);
        let cosmic_buf = CosmicBuffer::new(&mut font_sys, metrics);
        let mut s = Self {
            buffer: Buffer::empty(Rect::new(0, 0, cols, rows)),
            rgb: RgbPixmap::new(1, 1), // temp
            font_sys,
            cosmic_buf,
            cache: SwashCache::new(),
            cw: 0,
            ch: 0,
            blink_tick: 0,
            blink_fast: false,
            blink_slow: false,
            dirty: HashSet::new(),
            cursor_visible: false,
            cur_pos: (0, 0),
        };
        s.recalc_cell_metrics();
        s.rgb = RgbPixmap::new(s.cw * cols as usize, s.ch * rows as usize);
        let _ = s.clear();
        s
    }

    fn recalc_cell_metrics(&mut self) {
        let m = self.cosmic_buf.metrics().font_size;
        let line = &mut self.cosmic_buf.lines[0];
        line.set_text(
            "█",
            LineEnding::None,
            AttrsList::new(&Attrs::new().family(Family::Monospace)),
        );
        line.layout(&mut self.font_sys, m, None, Wrap::None, None, 1);
        let mut layout_runs = self.cosmic_buf.layout_runs();
        let glyph = layout_runs.next().unwrap().glyphs[0].physical((0., 0.), 1.0);
        let bbox = self
            .cache
            .get_image(&mut self.font_sys, glyph.cache_key)
            .unwrap()
            .as_ref()
            .placement;
        self.cw = bbox.width as usize;
        self.ch = bbox.height as usize;
        self.cosmic_buf.set_size(
            &mut self.font_sys,
            Some(self.cw as f32),
            Some(self.ch as f32),
        );
    }

    /* ----- drawing helpers (render one cell) ----- */

    fn draw_cell(&mut self, x: u16, y: u16) {
        let cell = self.buffer.cell(Position::new(x, y)).unwrap();
        let mut fg = cell.fg;
        let bg = cell.bg;
        if cell.modifier.contains(Modifier::HIDDEN) {
            fg = bg;
        }

        let (mut fg_rgb, mut bg_rgb) = if cell.modifier.contains(Modifier::REVERSED) {
            (rat_to_rgb(&bg, false), rat_to_rgb(&fg, true))
        } else {
            (rat_to_rgb(&fg, true), rat_to_rgb(&bg, false))
        };
        if cell.modifier.contains(Modifier::DIM) {
            fg_rgb = dim_rgb(fg_rgb);
            bg_rgb = dim_rgb(bg_rgb);
        }

        /* fill background rectangle */
        let ox = x as usize * self.cw;
        let oy = y as usize * self.ch;
        for yy in 0..self.ch {
            for xx in 0..self.cw {
                self.rgb.put_pixel(ox + xx, oy + yy, bg_rgb);
            }
        }

        /* glyph shaping & blit */
        let mut text = cell.symbol().to_string();
        if cell.modifier.contains(Modifier::CROSSED_OUT) {
            text = text.chars().flat_map(|c| [c, '\u{0336}']).collect();
        }
        if cell.modifier.contains(Modifier::UNDERLINED) {
            text = text.chars().flat_map(|c| [c, '\u{0332}']).collect();
        }

        if cell
            .modifier
            .intersects(Modifier::SLOW_BLINK | Modifier::RAPID_BLINK)
        {
            self.dirty.insert((x, y));
            let blink_state = if cell.modifier.contains(Modifier::RAPID_BLINK) {
                self.blink_fast
            } else {
                self.blink_slow
            };
            if blink_state {
                fg_rgb = bg_rgb;
            }
        }

        let mut attrs = Attrs::new().family(Family::Monospace);
        if cell.modifier.contains(Modifier::BOLD) {
            attrs = attrs.weight(Weight::BOLD);
        }
        if cell.modifier.contains(Modifier::ITALIC) {
            attrs = attrs.cache_key_flags(CacheKeyFlags::FAKE_ITALIC);
        }

        let m = self.cosmic_buf.metrics().font_size;
        let line = &mut self.cosmic_buf.lines[0];
        line.set_text(&text, LineEnding::None, AttrsList::new(&attrs));
        line.layout(&mut self.font_sys, m, None, Wrap::None, None, 1);

        for run in self.cosmic_buf.layout_runs() {
            for g in run.glyphs.iter() {
                let pg = g.physical((0., 0.), 1.0);
                if let Some(img) = self.cache.get_image(&mut self.font_sys, pg.cache_key) {
                    let px = pg.x + img.placement.left;
                    let py = run.line_y as i32 + pg.y - img.placement.top;
                    let mut idx = 0;
                    for yy in 0..img.placement.height {
                        for xx in 0..img.placement.width {
                            let rx = ox as i32 + px + xx as i32;
                            let ry = oy as i32 + py + yy as i32;
                            if rx >= 0 && ry >= 0 {
                                let blended = blend_rgba(
                                    [fg_rgb[0], fg_rgb[1], fg_rgb[2], img.data[idx]],
                                    [bg_rgb[0], bg_rgb[1], bg_rgb[2], 255],
                                );
                                self.rgb.put_pixel(rx as usize, ry as usize, blended);
                            }
                            idx += 1;
                        }
                    }
                }
            }
        }
    }

    /* full-redraw helper */
    fn redraw_all(&mut self) {
        self.dirty.clear();
        for y in 0..self.buffer.area.height {
            for x in 0..self.buffer.area.width {
                self.draw_cell(x, y);
            }
        }
    }
}

impl Backend for SoftBackend {
    type Error = io::Error;

    fn draw<'a, I>(&mut self, iter: I) -> io::Result<()>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        /* update blink state every call (~1 frame) */
        self.blink_tick = (self.blink_tick + 1) % 200;
        self.blink_fast = (self.blink_tick % 25) < 4; // 10 Hz
        self.blink_slow = (self.blink_tick % 100) < 6; // 2 Hz

        for (x, y, c) in iter {
            self.buffer[(x, y)] = c.clone();
            self.draw_cell(x, y);
        }
        /* re-draw dirty blink cells */
        let dirty_positions: Vec<(u16, u16)> = self.dirty.iter().copied().collect();
        for (x, y) in dirty_positions {
            self.draw_cell(x, y);
        }
        Ok(())
    }

    fn hide_cursor(&mut self) -> io::Result<()> {
        self.cursor_visible = false;
        Ok(())
    }
    fn show_cursor(&mut self) -> io::Result<()> {
        self.cursor_visible = true;
        Ok(())
    }

    fn get_cursor_position(&mut self) -> io::Result<Position> {
        Ok(self.cur_pos.into())
    }
    fn set_cursor_position<P: Into<Position>>(&mut self, p: P) -> io::Result<()> {
        self.cur_pos = p.into().into();
        Ok(())
    }

    fn clear(&mut self) -> io::Result<()> {
        self.buffer.reset();
        self.rgb.fill(rat_to_rgb(&Cell::EMPTY.bg, false));
        Ok(())
    }

    fn clear_region(&mut self, region: ClearType) -> io::Result<()> {
        match region {
            ClearType::All => self.clear(),
            ClearType::AfterCursor => {
                // Clear from cursor position to end
                let pos = self.get_cursor_position()?;
                let area = self.buffer.area;
                for y in pos.y..area.bottom() {
                    let start_x = if y == pos.y { pos.x } else { area.left() };
                    for x in start_x..area.right() {
                        if let Some(cell) = self.buffer.cell_mut((x, y)) {
                            *cell = Cell::default();
                        }
                        let ox = x as usize * self.cw;
                        let oy = y as usize * self.ch;
                        for yy in 0..self.ch {
                            for xx in 0..self.cw {
                                self.rgb.put_pixel(
                                    ox + xx,
                                    oy + yy,
                                    rat_to_rgb(&Cell::EMPTY.bg, false),
                                );
                            }
                        }
                    }
                }
                Ok(())
            }
            ClearType::BeforeCursor => {
                // Clear from start to cursor position
                let pos = self.get_cursor_position()?;
                let area = self.buffer.area;
                for y in area.top()..=pos.y {
                    let end_x = if y == pos.y { pos.x } else { area.right() };
                    for x in area.left()..end_x {
                        if let Some(cell) = self.buffer.cell_mut((x, y)) {
                            *cell = Cell::default();
                        }
                        let ox = x as usize * self.cw;
                        let oy = y as usize * self.ch;
                        for yy in 0..self.ch {
                            for xx in 0..self.cw {
                                self.rgb.put_pixel(
                                    ox + xx,
                                    oy + yy,
                                    rat_to_rgb(&Cell::EMPTY.bg, false),
                                );
                            }
                        }
                    }
                }
                Ok(())
            }
            ClearType::CurrentLine => {
                // Clear current line
                let pos = self.get_cursor_position()?;
                let area = self.buffer.area;
                for x in area.left()..area.right() {
                    if let Some(cell) = self.buffer.cell_mut((x, pos.y)) {
                        *cell = Cell::default();
                    }
                    let ox = x as usize * self.cw;
                    let oy = pos.y as usize * self.ch;
                    for yy in 0..self.ch {
                        for xx in 0..self.cw {
                            self.rgb.put_pixel(
                                ox + xx,
                                oy + yy,
                                rat_to_rgb(&Cell::EMPTY.bg, false),
                            );
                        }
                    }
                }
                Ok(())
            }
            ClearType::UntilNewLine => {
                // Clear from cursor to end of line
                let pos = self.get_cursor_position()?;
                let area = self.buffer.area;
                for x in pos.x..area.right() {
                    if let Some(cell) = self.buffer.cell_mut((x, pos.y)) {
                        *cell = Cell::default();
                    }
                    let ox = x as usize * self.cw;
                    let oy = pos.y as usize * self.ch;
                    for yy in 0..self.ch {
                        for xx in 0..self.cw {
                            self.rgb.put_pixel(
                                ox + xx,
                                oy + yy,
                                rat_to_rgb(&Cell::EMPTY.bg, false),
                            );
                        }
                    }
                }
                Ok(())
            }
        }
    }

    fn size(&self) -> io::Result<Size> {
        Ok(self.buffer.area.as_size())
    }

    fn window_size(&mut self) -> io::Result<WindowSize> {
        Ok(WindowSize {
            columns_rows: self.size()?.into(),
            pixels: Size {
                width: self.pw() as u16,
                height: self.ph() as u16,
            },
        })
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
