/// A contiguous RGB frame-buffer. One pixel = three bytes (R,G,B).
#[derive(Debug, Clone)]
pub struct RgbPixmap {
    w: usize,
    h: usize,
    data: Vec<u8>,
}

impl RgbPixmap {
    /// Allocate an empty pixmap initialised to black.
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            data: vec![0; w * h * 3],
        }
    }

    /// Convert to RGBA (alpha=255).
    pub fn to_rgba(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.data.len() / 3 * 4);
        for chunk in self.data.chunks_exact(3) {
            v.extend_from_slice(chunk);
            v.push(255);
        }
        v
    }

    /// Set a single pixel.
    #[inline]
    pub fn put_pixel(&mut self, x: usize, y: usize, rgb: [u8; 3]) {
        debug_assert!(x < self.w && y < self.h);
        let idx = 3 * (y * self.w + x);
        self.data[idx..idx + 3].copy_from_slice(&rgb);
    }

    /// Read a pixel.
    #[inline]
    pub fn get_pixel(&self, x: usize, y: usize) -> [u8; 3] {
        debug_assert!(x < self.w && y < self.h);
        let idx = 3 * (y * self.w + x);
        self.data[idx..idx + 3].try_into().unwrap()
    }

    /// Fill the entire buffer.
    pub fn fill(&mut self, rgb: [u8; 3]) {
        for chunk in self.data.chunks_mut(3) {
            chunk.copy_from_slice(&rgb);
        }
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }
    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}
