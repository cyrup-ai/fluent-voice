//! src/codec.rs
//! -------------------------------------------------------------------------
//! Decode any common audio file, loudness-normalise, resample to 24 kHz mono
//! and run it through Facebook EnCodec (24 kHz) – all on the chosen device.

#[cfg(not(any(feature = "encodec", feature = "mimi", feature = "snac")))]
use anyhow;
use anyhow::Result;
#[cfg(any(feature = "encodec", feature = "mimi", feature = "snac"))]
use anyhow::{Context, anyhow};
use candle_core::{Device, Tensor};

#[cfg(any(feature = "encodec", feature = "mimi", feature = "snac"))]
use crate::audio::{SAMPLE_RATE, normalize_loudness, to_24k_mono};
#[cfg(any(feature = "encodec", feature = "mimi", feature = "snac"))]
use crate::model::load_encodec;
#[cfg(any(feature = "encodec", feature = "mimi", feature = "snac"))]
use symphonia::core::{
    audio::{AudioBufferRef, Signal},
    codecs::DecoderOptions,
    conv::FromSample,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
    sample::Sample,
};

// pcm_decode import removed - unused

// ----------------------------------------------------------------------------
// One-time EnCodec loader (mmap'ed safetensors → Model on CPU / CUDA / Metal).
// ----------------------------------------------------------------------------
// static ENCODEC_MODEL: OnceCell<encodec::Model> = OnceCell::new();

// fn load_encodec(device: &Device) -> Result<&'static encodec::Model> {
//     ENCODEC_MODEL.get_or_try_init(|| {
//         let api = Api::new().context("initialising HF hub")?;
//         let weights = api
//             .model("facebook/encodec_24khz".to_string())
//             .get("model.safetensors")
//             .context("downloading EnCodec weights")?;
//         let vb = unsafe {
//             VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)
//                 .context("mmap EnCodec weights")?
//         };
//         encodec::Model::new(&encodec::Config::default(), vb).context("building EnCodec model")
//     })
// }

// ----------------------------------------------------------------------------
// Public helper
// ----------------------------------------------------------------------------
#[cfg(any(feature = "encodec", feature = "mimi", feature = "snac"))]
pub fn encode_wav(path: &str, device: &Device, compress: bool) -> Result<Tensor> {
    // 1) -------- Decode ----------------------------------------------------
    let file = std::fs::File::open(path).with_context(|| format!("open {path}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let probed = symphonia::default::get_probe().format(
        &Hint::new(),
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;

    // ── track / codec-params (borrow only inside this block) ─────────────
    let (track_id, params) = {
        let track = format
            .default_track()
            .ok_or_else(|| anyhow!("no audio track"))?;
        (track.id, track.codec_params.clone())
    };

    // sample-rate & channel count
    let sr_in = params.sample_rate.ok_or_else(|| anyhow!("unknown sr"))?;
    let channels = params.channels.map(|c| c.count()).unwrap_or(1);

    // now it's safe to mutably borrow `format`
    let mut decoder = symphonia::default::get_codecs().make(&params, &DecoderOptions::default())?;

    let mut pcm = Vec::<f32>::new();
    while let Ok(pkt) = format.next_packet() {
        if pkt.track_id() != track_id {
            continue;
        }
        let decoded = decoder.decode(&pkt)?;
        match decoded {
            AudioBufferRef::F32(buf) => pcm.extend(buf.chan(0)),
            AudioBufferRef::F64(buf) => extend::<f64>(&mut pcm, buf.chan(0)),
            AudioBufferRef::U8(buf) => extend::<u8>(&mut pcm, buf.chan(0)),
            AudioBufferRef::U16(buf) => extend::<u16>(&mut pcm, buf.chan(0)),
            AudioBufferRef::U24(_) => continue, // Skip 24-bit formats for now
            AudioBufferRef::U32(buf) => extend::<u32>(&mut pcm, buf.chan(0)),
            AudioBufferRef::S8(buf) => extend::<i8>(&mut pcm, buf.chan(0)),
            AudioBufferRef::S16(buf) => extend::<i16>(&mut pcm, buf.chan(0)),
            AudioBufferRef::S24(_) => continue, // Skip 24-bit formats for now
            AudioBufferRef::S32(buf) => extend::<i32>(&mut pcm, buf.chan(0)),
        }
    }

    // 2) -------- Down-mix + resample --------------------------------------
    let pcm = to_24k_mono(pcm, sr_in, channels).context("stereo→mono / resample")?;

    // 3) -------- Loudness --------------------------------------------------
    let wav = Tensor::from_slice(&pcm, (pcm.len(),), device)?;
    let wav =
        normalize_loudness(&wav, SAMPLE_RATE as u32, compress).context("BS.1770 normalisation")?;

    // 4) -------- EnCodec ---------------------------------------------------
    let model = load_encodec(device)?;
    let wav_b1t = wav.unsqueeze(0)?.unsqueeze(0)?; // [1,1,T]
    let codes = model.encode(&wav_b1t)?; // [1,B,T]
    let codes = codes.squeeze(0)?.transpose(0, 1)?; // [T,B]

    Ok(codes)
}

#[cfg(not(any(feature = "encodec", feature = "mimi", feature = "snac")))]
pub fn encode_wav(_path: &str, _device: &Device, _compress: bool) -> Result<Tensor> {
    Err(anyhow::anyhow!(
        "WAV encoding requires encodec, mimi, or snac features"
    ))
}

// ------------ tiny generic helper ------------------------------------------
#[cfg(any(feature = "encodec", feature = "mimi", feature = "snac"))]
fn extend<T>(dst: &mut Vec<f32>, src: &[T])
where
    T: Sample,
    f32: FromSample<T>,
{
    dst.extend(src.iter().map(|&s| f32::from_sample(s)));
}
