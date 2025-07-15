//! Symphonia loader  → f32 mono PCM

use anyhow::Result;

#[cfg(any(feature = "encodec", feature = "mimi", feature = "snac"))]
use symphonia::{
    core::{
        audio::{AudioBufferRef, Signal},
        codecs,
        conv::FromSample,
        formats::FormatOptions,
        io::MediaSourceStream,
        meta::MetadataOptions,
        probe::Hint,
        sample::Sample,
    },
    default,
};

#[cfg(any(feature = "encodec", feature = "mimi", feature = "snac"))]
pub fn load(path: &str) -> Result<(Vec<f32>, u32)> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path).extension() {
        let ext_str = ext.to_string_lossy();
        hint.with_extension(&ext_str);
    }

    let probed = default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;

    // Extract track information before starting packet processing
    // to avoid overlapping borrows
    let track_id;
    let codec_params;
    let sr;
    {
        // first audio track
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != codecs::CODEC_TYPE_NULL)
            .context("no supported audio track found")?;

        track_id = track.id;
        codec_params = track.codec_params.clone();
        sr = codec_params.sample_rate.context("unknown sr")?;
    }

    let mut decoder = default::get_codecs().make(&codec_params, &Default::default())?;
    let mut pcm = Vec::<f32>::new();

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet)? {
            AudioBufferRef::F32(buf) => pcm.extend(buf.chan(0)),
            AudioBufferRef::F64(buf) => extend::<f64>(&mut pcm, &buf.chan(0)),
            AudioBufferRef::U8(buf) => extend::<u8>(&mut pcm, &buf.chan(0)),
            AudioBufferRef::U16(buf) => extend::<u16>(&mut pcm, &buf.chan(0)),
            AudioBufferRef::U32(buf) => extend::<u32>(&mut pcm, &buf.chan(0)),
            AudioBufferRef::S8(buf) => extend::<i8>(&mut pcm, &buf.chan(0)),
            AudioBufferRef::S16(buf) => extend::<i16>(&mut pcm, &buf.chan(0)),
            AudioBufferRef::S32(buf) => extend::<i32>(&mut pcm, &buf.chan(0)),
            _ => unreachable!("unsupported buffer type"),
        }
    }
    Ok((pcm, sr))
}

#[cfg(any(feature = "encodec", feature = "mimi", feature = "snac"))]
fn extend<T>(dst: &mut Vec<f32>, src: &[T])
where
    T: Sample,
    f32: FromSample<T>,
{
    dst.extend(src.iter().map(|s| f32::from_sample(*s)));
}

#[cfg(not(any(feature = "encodec", feature = "mimi", feature = "snac")))]
pub fn load(_path: &str) -> Result<(Vec<f32>, u32)> {
    Err(anyhow::anyhow!(
        "PCM loading requires encodec, mimi, or snac features"
    ))
}
