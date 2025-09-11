//---
// path: potter-core/src/wakeword/build.rs
//---
use std::{collections::HashMap, io::BufReader, path::Path, sync::Arc};

#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::cmp::Ordering;

use crate::{
    WakewordRef,
    kfc::{KfcAverager, KfcWavFileExtractor},
};

/// Errors returned while constructing a [`WakewordRef`].
#[derive(Debug, thiserror::Error)]
pub enum BuilderError {
    #[error("file not found: {0}")]
    NotFound(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("wav: {0}")]
    Wav(String),
    #[error("no samples provided")]
    Empty,
    #[error("kfc: {0}")]
    Kfc(String),
}

type SampleMap = HashMap<String, Arc<Vec<u8>>>;

/// Build a [`WakewordRef`] from **any** iterator that yields `(name, wav_bytes)`.
///
/// ```no_run
/// # use potter_core::wakeword::build::{build_from_iter, BuilderError};
/// # fn main() -> Result<(), BuilderError> {
/// let files = vec![
///     ("positron_1.wav", std::fs::read("positron_1.wav")?),
///     ("positron_2.wav", std::fs::read("positron_2.wav")?),
/// ];
/// let wake = build_from_iter(
///     "positron".into(),
///     None, None,
///     files.into_iter(),
///     13,
/// )?;
/// # Ok(()) }
/// ```
pub fn build_from_iter<I, S, B>(
    name: String,
    threshold: Option<f32>,
    avg_threshold: Option<f32>,
    samples: I,
    kfc_size: u16,
) -> Result<WakewordRef, BuilderError>
where
    I: IntoIterator<Item = (S, B)>,
    S: AsRef<str>,
    B: AsRef<[u8]>,
{
    let mut sample_map: SampleMap = HashMap::new();
    for (n, buf) in samples {
        sample_map.insert(n.as_ref().to_owned(), Arc::new(buf.as_ref().to_owned()));
    }
    if sample_map.is_empty() {
        return Err(BuilderError::Empty);
    }

    // ---------- KFC extraction (parallel if rayon available) -----------
    let mut rms_levels = Vec::<f32>::with_capacity(sample_map.len());

    #[cfg(feature = "rayon")]
    let samples_features: HashMap<_, _> = {
        let par_iter = sample_map.par_iter();
        par_iter
            .map(|(name, bytes)| {
                let mut rms = 0.0;
                let feats = KfcWavFileExtractor::compute_kfcs(
                    BufReader::new(&bytes[..]),
                    &mut rms,
                    kfc_size, // Ensure correct type for kfc_size
                )
                .map_err(|e| BuilderError::Kfc(e.to_string()))?;
                Ok::<_, BuilderError>((name.clone(), feats, rms))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|(n, f, r)| {
                rms_levels.push(r);
                (n, f)
            })
            .collect()
    };

    #[cfg(not(feature = "rayon"))]
    let samples_features: HashMap<_, _> = {
        sample_map
            .iter()
            .map(|(name, bytes)| {
                let mut rms = 0.0;
                let feats = KfcWavFileExtractor::compute_kfcs(
                    BufReader::new(&bytes[..]),
                    &mut rms,
                    kfc_size, // Ensure correct type for kfc_size
                )
                .map_err(|e| BuilderError::Kfc(e.to_string()))?;
                rms_levels.push(rms);
                Ok::<_, BuilderError>((name.clone(), feats))
            })
            .collect::<Result<HashMap<_, _>, _>>()?
    };

    // ---------- Median RMS -----------------------------------------------
    rms_levels.sort_by(|a, b| a.total_cmp(b));
    let rms_level = rms_levels[rms_levels.len() / 2];

    // ---------- Average template -----------------------------------------
    let avg = compute_avg(&samples_features);

    WakewordRef::new(
        name,
        threshold,
        avg_threshold,
        avg,
        rms_level,
        samples_features,
    )
    .map_err(BuilderError::Kfc)
}

/* ----------------------------------------------------------------------- */

/// Convenience: build from a **list of file paths**.
pub fn build_from_files(
    name: String,
    threshold: Option<f32>,
    avg_threshold: Option<f32>,
    paths: &[impl AsRef<Path>],
    kfc_size: u16,
) -> Result<WakewordRef, BuilderError> {
    let iter = paths.iter().map(|p| {
        let path = p.as_ref();
        let fname = path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| BuilderError::NotFound(path.display().to_string()))?
            .to_owned();
        let buf = std::fs::read(path).map_err(BuilderError::Io)?;
        Ok::<_, BuilderError>((fname, buf))
    });
    build_from_iter(
        name,
        threshold,
        avg_threshold,
        iter.collect::<Result<Vec<_>, _>>()?,
        kfc_size,
    )
}

/// Compute the DTW-aligned average template (if >1 sample).
fn compute_avg(samples: &HashMap<String, Vec<Vec<f32>>>) -> Option<Vec<Vec<f32>>> {
    if samples.len() <= 1 {
        return None;
    }
    let mut sorted: Vec<_> = samples.values().collect();
    sorted.sort_by(|a, b| {
        let ord = b.len().cmp(&a.len());
        if ord == Ordering::Equal {
            std::ptr::addr_of!(*a).cmp(&std::ptr::addr_of!(*b))
        } else {
            ord
        }
    });
    let refs: Vec<_> = sorted.into_iter().cloned().collect();
    KfcAverager::average(&refs).ok()
}
