use ratatui::style::Color as RatColor;

/// Convert a Ratatui [`Color`] into an RGB triplet.
#[inline]
pub fn rat_to_rgb(rat_col: &RatColor, is_fg: bool) -> [u8; 3] {
    match rat_col {
        RatColor::Reset => {
            if is_fg {
                [204, 204, 255] // default foreground
            } else {
                [15, 15, 112] // default background
            }
        }
        RatColor::Black => [0, 0, 0],
        RatColor::Red => [139, 0, 0],
        RatColor::Green => [0, 100, 0],
        RatColor::Yellow => [255, 215, 0],
        RatColor::Blue => [0, 0, 139],
        RatColor::Magenta => [99, 9, 99],
        RatColor::Cyan => [0, 0, 255],
        RatColor::Gray => [128, 128, 128],
        RatColor::DarkGray => [64, 64, 64],
        RatColor::LightRed => [255, 0, 0],
        RatColor::LightGreen => [0, 255, 0],
        RatColor::LightBlue => [173, 216, 230],
        RatColor::LightYellow => [255, 255, 224],
        RatColor::LightMagenta => [139, 0, 139],
        RatColor::LightCyan => [224, 255, 255],
        RatColor::White => [255, 255, 255],
        RatColor::Indexed(i) => {
            let i = *i as u8;
            [i.wrapping_mul(i), i.wrapping_add(i), i]
        }
        RatColor::Rgb(r, g, b) => [*r, *g, *b],
    }
}

/// Alpha-blend two RGBA pixels (premultiplied alpha not required).
#[inline]
pub fn blend_rgba(fg: [u8; 4], bg: [u8; 4]) -> [u8; 3] {
    let fg_a = fg[3] as f32 / 255.0;
    let bg_a = bg[3] as f32 / 255.0;
    let out_a = fg_a + bg_a * (1.0 - fg_a);
    if out_a == 0.0 {
        return [0, 0, 0];
    }
    let blend = |f, b| ((f as f32 * fg_a + b as f32 * bg_a * (1.0 - fg_a)) / out_a).round() as u8;
    [
        blend(fg[0], bg[0]),
        blend(fg[1], bg[1]),
        blend(fg[2], bg[2]),
    ]
}

/// Dim an RGB colour by ~70 % (used for [`Modifier::DIM`]).
#[inline]
pub fn dim_rgb(rgb: [u8; 3]) -> [u8; 3] {
    const FACTOR: u32 = 77; // ≈ 255 × 0.30
    [
        ((rgb[0] as u32 * FACTOR + 127) / 255) as u8,
        ((rgb[1] as u32 * FACTOR + 127) / 255) as u8,
        ((rgb[2] as u32 * FACTOR + 127) / 255) as u8,
    ]
}
