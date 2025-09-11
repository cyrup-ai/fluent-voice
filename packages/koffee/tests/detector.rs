//! High-level integration tests for the Kfc crate.
//
//  Run with `cargo test -p rustpotter --test detect`
//
//  – Removes a ton of boiler-plate by using tiny helpers and
//    a table-driven pattern for the many “same-thing with
//    different knobs” scenarios.
//  – Keeps every case its own `#[test]`, so they still show up
//    individually in `cargo test` output.
//

use koffee::{
    Kfc, KoffeeCandleDetection, ScoreMode,
    config::{
        AudioFmt, DetectorConfig, Endianness, FiltersConfig, KoffeeCandleConfig, SampleFormat,
        VADMode,
    },
    wakewords::{ModelType, WakewordLoad},
};
use rustpotter::{WakewordModelTrain, WakewordModelTrainOptions, WakewordRef};
use std::{fs::File, io::Read, path::PathBuf};

/* ───────────────────────────── helpers ────────────────────────────── */

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn p(rel: &str) -> String {
    root()
        .join(rel.trim_start_matches('/'))
        .to_string_lossy()
        .into_owned()
}

fn silence(sr: usize, secs: usize, bytes_per_sample: usize) -> Vec<u8> {
    vec![0; sr * secs * bytes_per_sample]
}

/// Read a *mono* i16 WAV, strip the header and apply a gain factor.
/// **Assumes** little-endian 16-bit PCM.
fn wav_data(path: &str, gain: f32) -> Vec<u8> {
    let mut buf = Vec::new();
    File::open(path).unwrap().read_to_end(&mut buf).unwrap();
    // Skip 44-byte RIFF header.
    buf.drain(..44);
    buf.chunks_exact(2)
        .flat_map(|b| {
            let s = i16::from_le_bytes([b[0], b[1]]);
            let g = (s as f32 * gain)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            g.to_le_bytes()
        })
        .collect()
}

/* ───────────────────────────── core runner ────────────────────────── */

fn run_bytes(
    mut cfg: KoffeeCandleConfig,
    model: &str,
    bytes: Vec<u8>,
) -> Vec<KoffeeCandleDetection> {
    let mut potter = Kfc::new(&cfg).unwrap();
    potter.add_wakeword_from_file("ww", &p(model)).unwrap();

    bytes
        .chunks_exact(potter.get_bytes_per_frame())
        .filter_map(|b| potter.process_bytes(b))
        .collect()
}

fn run_samples(mut cfg: KoffeeCandleConfig, model: &str, wav: &str) -> Vec<KoffeeCandleDetection> {
    // point Kfc at the WAV format so we can feed samples directly
    let r = hound::WavReader::open(p(wav)).unwrap();
    let spec = r.spec();
    cfg.fmt = AudioFmt {
        sample_rate: spec.sample_rate,
        channels: spec.channels as u32,
        sample_format: match spec.sample_format {
            hound::SampleFormat::Float => SampleFormat::F32,
            hound::SampleFormat::Int => match spec.bits_per_sample {
                16 => SampleFormat::I16,
                32 => SampleFormat::I32,
                _ => SampleFormat::I16, // default fallback
            },
        },
        endianness: Endianness::Little, // WAV is typically little-endian
    };

    let samples: Vec<f32> = r.into_samples::<f32>().map(|x| x.unwrap()).collect();

    let mut potter = Kfc::new(&cfg).unwrap();
    potter.add_wakeword_from_file("ww", &p(model)).unwrap();

    samples
        .chunks_exact(potter.get_samples_per_frame())
        .filter_map(|c| potter.process_samples(c.to_vec()))
        .collect()
}

/* ───────────────────────────── macro magic ────────────────────────── */

macro_rules! cfg_base {
    () => {{
        let mut c = KoffeeCandleConfig::default();
        c.filters.band_pass.enabled = false;
        c.filters.gain_normalizer.enabled = false;
        c
    }};
}

macro_rules! assert_det {
    ($det:expr, $avg:expr, $score:expr) => {{
        approx::assert_abs_diff_eq!($det.avg_score, $avg, epsilon = 1e-5);
        approx::assert_abs_diff_eq!($det.score, $score, epsilon = 1e-5);
    }};
}

/* ───────────────────────────── crispy tests ───────────────────────── */

#[test]
fn v2_file_max() {
    let mut c = cfg_base!();
    c.detector.avg_threshold = 0.2;
    c.detector.threshold = 0.5;
    c.detector.score_mode = ScoreMode::Max;

    let det = run_samples(
        c,
        "/tests/resources/oye_casa_g_v2.rpw",
        "/tests/resources/oye_casa_g_1.wav",
    );
    assert_eq!(det.len(), 2);
    assert_det!(det[0], 0.6495044, 0.7310586);
    assert_det!(det[1], 0.5804737, 0.721843);
}

#[test]
fn v1_max_vs_median_vs_avg() {
    for (mode, exp) in &[
        (ScoreMode::Max, (0.7310586, 0.6495044)),
        (ScoreMode::Median, (0.60123634, 0.64608675)),
        (ScoreMode::Average, (0.60458726, 0.64608675)),
    ] {
        let mut c = cfg_base!();
        c.detector.avg_threshold = 0.2;
        c.detector.threshold = 0.5;
        c.detector.score_mode = *mode;

        let det = run_samples(
            c,
            "/tests/resources/oye_casa_g.rpw",
            "/tests/resources/oye_casa_g_1.wav",
        );
        assert_eq!(det.len(), 2);
        assert_det!(det[0], exp.1, exp.0);
    }
}

#[test]
fn max_with_vad() {
    let mut c = cfg_base!();
    c.detector.avg_threshold = 0.2;
    c.detector.threshold = 0.5;
    c.detector.score_mode = ScoreMode::Max;
    c.detector.vad_mode = Some(VADMode::Easy);

    let det = run_samples(
        c,
        "/tests/resources/oye_casa_g.rpw",
        "/tests/resources/oye_casa_g_1.wav",
    );
    assert_eq!(det.len(), 2);
    assert_det!(det[0], 0.6495044, 0.7310586);
}

#[test]
fn ignore_alexa() {
    let mut c = cfg_base!();
    c.detector.threshold = 0.45;
    c.detector.score_mode = ScoreMode::Max;
    c.detector.min_scores = 0;

    assert!(
        run_samples(
            c.clone(),
            "/tests/resources/alexa.rpw",
            "/tests/resources/oye_casa_g_1.wav"
        )
        .is_empty()
    );

    // with filters enabled
    c.filters.band_pass.enabled = true;
    c.filters.gain_normalizer.enabled = true;
    assert!(
        run_samples(
            c,
            "/tests/resources/alexa.rpw",
            "/tests/resources/oye_casa_g_1.wav"
        )
        .is_empty()
    );
}

#[test]
fn gain_and_bandpass_filters() {
    let mut c = cfg_base!();
    c.detector.threshold = 0.5;
    c.filters.gain_normalizer.enabled = true;
    c.filters.band_pass.enabled = true;
    c.filters.band_pass.low_cutoff = 80.;
    c.filters.band_pass.high_cutoff = 500.;
    c.detector.score_mode = ScoreMode::Median;

    // build byte stream:  silence – oye  – silence – oye – silence
    let sr = 16_000;
    let bytes = {
        let mut v = silence(sr, 5, 2);
        v.extend(wav_data(&p("/tests/resources/oye_casa_g_1.wav"), 0.2));
        v.extend(silence(sr, 5, 2));
        v.extend(wav_data(&p("/tests/resources/oye_casa_g_2.wav"), 5.0));
        v.extend(silence(sr, 5, 2));
        v
    };

    let det = run_bytes(c, "/tests/resources/oye_casa_g.rpw", bytes);
    assert_eq!(det.len(), 2);
    approx::assert_abs_diff_eq!(det[0].score, 0.5775406, epsilon = 1e-5);
    approx::assert_abs_diff_eq!(det[1].score, 0.5828697, epsilon = 1e-5);
}

/* ───────────────────────────── tiny smoke for NN model ─────────────── */

#[test]
fn nn_model_ok_casa() {
    let mut c = KoffeeCandleConfig::default();
    c.detector.avg_threshold = 0.;

    let det = run_samples(
        c,
        "/tests/resources/ok_casa-tiny.rpw",
        "/tests/resources/ok_casa.wav",
    );

    assert_eq!(det.len(), 1);
    let d = &det[0];
    assert_eq!(d.counter, 34);
    approx::assert_abs_diff_eq!(d.score, 0.9997649, epsilon = 1e-5);
    approx::assert_abs_diff_eq!(d.scores["ok_casa"], 3.7506533, epsilon = 1e-5);
}

/* ───────────────────────────── misc API tests ──────────────────────── */

#[test]
fn remove_wakewords() {
    let cfg = KoffeeCandleConfig::default();
    let mut potter = Kfc::new(&cfg).unwrap();
    potter
        .add_wakeword_from_file("key", &p("/tests/resources/ok_casa-tiny.rpw"))
        .unwrap();
    assert!(potter.remove_wakeword("key"));
    assert!(potter.remove_wakewords());
}

/* ───────────────────────────── quick train smoke ───────────────────── */

#[test]
fn train_smoke() {
    let train = p("/tests/resources/train");
    let test = p("/tests/resources/test");
    let opts = WakewordModelTrainOptions::new(ModelType::Medium, 0.027, 1, 1, 16); // 1 epoch for speed

    let model = rustpotter::WakewordModel::train_from_dirs(opts, train, test, None).unwrap();
    assert_eq!(model.labels.len(), 2);
}
