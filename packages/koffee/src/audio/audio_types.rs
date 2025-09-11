//! Audio sample-format primitives used throughout Kfc.
//
//  • All variants document their size explicitly in the name (`I16`, `F32`, …).
//  • Helpers convert bits ↔︎ enum and provide safe fall-backs.
//  • `Sample` trait is blanket-implemented via a macro for every scalar type
//    we support; new types can be added with one extra `with_sample_type!` call.

/* ─────────────────────────────── ENUMS ─────────────────────────────── */

/// PCM sample encoding.
#[derive(Clone, Copy, Debug)]
pub enum SampleFormat {
    /// 8-bit signed integer samples.
    I8,
    /// 16-bit signed integer samples.
    I16,
    /// 32-bit signed integer samples.
    I32,
    /// 32-bit floating point samples.
    F32,
}

impl SampleFormat {
    /// Bits per scalar.
    #[inline]
    pub const fn bits(self) -> u16 {
        match self {
            Self::I8 => 8,
            Self::I16 => 16,
            Self::I32 | Self::F32 => 32,
        }
    }
    /// Bytes per scalar.
    #[inline]
    pub const fn bytes(self) -> u16 {
        self.bits() / 8
    }

    /// Bytes per sample (alias for bytes)
    #[inline]
    pub const fn get_bytes_per_sample(self) -> u16 {
        self.bytes()
    }

    /// Integer format from bit size (`8/16/32`).
    pub const fn int_of_size(bits: u16) -> Option<Self> {
        match bits {
            8 => Some(Self::I8),
            16 => Some(Self::I16),
            32 => Some(Self::I32),
            _ => None,
        }
    }
    /// Float format from bit size (`32`).
    pub const fn float_of_size(bits: u16) -> Option<Self> {
        if bits == 32 { Some(Self::F32) } else { None }
    }
}

#[cfg(feature = "display")]
impl fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::F32 => "f32",
        })
    }
}

/// Endianness of a byte stream.
#[derive(Clone, Debug)]
pub enum Endianness {
    /// Big-endian byte ordering.
    Big,
    /// Little-endian byte ordering.
    Little,
    /// Native system byte ordering.
    Native,
}

/* ──────────────────────────── TRAIT CORE ───────────────────────────── */

/// A scalar type accepted by the DSP pipeline.
///
/// *All* conversions are inline and panic-free.
pub trait Sample: Copy + PartialOrd + Send + 'static {
    /// The format of this sample type.
    const FORMAT: SampleFormat;
    /// Returns the zero value for this sample type.
    fn zero() -> Self;
    /// Returns the number of bytes per sample.
    fn bytes() -> usize {
        Self::FORMAT.bytes() as usize
    }
    /// Creates a sample from little-endian bytes.
    fn from_le(b: &[u8]) -> Self;
    /// Creates a sample from big-endian bytes.
    fn from_be(b: &[u8]) -> Self;
    /// Creates a sample from native-endian bytes.
    fn from_ne(b: &[u8]) -> Self;
    /// Converts this sample to f32.
    fn into_f32(self) -> f32;

    // Additional methods needed by encoder
    /// Returns the byte size of this sample type.
    fn get_byte_size() -> usize {
        Self::FORMAT.bytes() as usize
    }
    /// Creates a sample from little-endian bytes (alias for from_le).
    fn from_le_bytes(b: &[u8]) -> Self {
        Self::from_le(b)
    }
    /// Creates a sample from big-endian bytes (alias for from_be).
    fn from_be_bytes(b: &[u8]) -> Self {
        Self::from_be(b)
    }
    /// Creates a sample from native-endian bytes (alias for from_ne).
    fn from_ne_bytes(b: &[u8]) -> Self {
        Self::from_ne(b)
    }
}

/* ───────────────────── blanket impl via macro ──────────────────────── */

macro_rules! with_sample_type {
    ($ty:ty, $variant:ident, $to_f32:expr, $from:ident, $zero:expr) => {
        impl Sample for $ty {
            const FORMAT: SampleFormat = SampleFormat::$variant;
            #[inline]
            fn zero() -> Self {
                $zero
            }
            #[inline]
            fn from_le(b: &[u8]) -> Self {
                match b.try_into() {
                    Ok(bytes) => <$ty>::from_le_bytes(bytes),
                    Err(_) => Self::zero(), // Safe fallback if byte slice is wrong size
                }
            }
            #[inline]
            fn from_be(b: &[u8]) -> Self {
                match b.try_into() {
                    Ok(bytes) => <$ty>::from_be_bytes(bytes),
                    Err(_) => Self::zero(), // Safe fallback if byte slice is wrong size
                }
            }
            #[inline]
            fn from_ne(b: &[u8]) -> Self {
                match b.try_into() {
                    Ok(bytes) => <$ty>::from_ne_bytes(bytes),
                    Err(_) => Self::zero(), // Safe fallback if byte slice is wrong size
                }
            }
            #[inline]
            fn into_f32(self) -> f32 {
                $to_f32(self)
            }
        }
    };
}

/* i8  */
with_sample_type!(i8, I8, |v: i8| v as f32 / i8::MAX as f32, from_le_bytes, 0);
/* i16 */
with_sample_type!(
    i16,
    I16,
    |v: i16| v as f32 / i16::MAX as f32,
    from_le_bytes,
    0
);
/* i32 */
with_sample_type!(
    i32,
    I32,
    |v: i32| v as f32 / i32::MAX as f32,
    from_le_bytes,
    0
);
/* f32 */
with_sample_type!(f32, F32, |v: f32| v, from_le_bytes, 0.0);
