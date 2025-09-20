pub mod format;

pub mod cpal;

pub type Matrix<T> = Vec<Vec<T>>;

#[derive(Debug, thiserror::Error)]
pub enum AudioDataError {
    #[error("Audio device unavailable")]
    DeviceUnavailable,

    #[error("Audio buffer overflow")]
    BufferOverflow,

    #[error("Invalid sample rate: {rate}")]
    InvalidSampleRate { rate: u32 },

    #[error("Audio device disconnected")]
    DeviceDisconnected,

    #[error("Channel receive error: {0}")]
    ChannelReceive(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Audio device error: {0}")]
    Device(#[from] cpal::AudioDeviceErrors),
}

pub trait DataSource<T> {
    fn recv(&mut self) -> Result<Matrix<T>, AudioDataError>;
}

/// separate a stream of alternating channels into a matrix of channel streams:
///   L R L R L R L R L R
/// becomes
///   L L L L L
///   R R R R R
pub fn stream_to_matrix<I, O>(
    stream: impl Iterator<Item = I>,
    channels: usize,
    norm: O,
) -> Matrix<O>
where
    I: Copy + Into<O>,
    O: Copy + std::ops::Div<Output = O>,
{
    let mut out = vec![vec![]; channels];
    let mut channel = 0;
    for sample in stream {
        out[channel].push(sample.into() / norm);
        channel = (channel + 1) % channels;
    }
    out
}
