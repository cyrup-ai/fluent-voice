//! fluent_voice/src/audio_stream.rs
//! --------------------------------
//! Audio stream output type

#[derive(Debug)]
pub struct AudioStream {
    pub pcm:         Vec<i16>,
    pub sample_rate: u32,
}