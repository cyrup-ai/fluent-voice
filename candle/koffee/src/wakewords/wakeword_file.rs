//---
// path: potter-core/src/wakeword/io.rs
//---
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use ciborium::{de, ser};
use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

use super::WakewordDetector;
use crate::ScoreMode;

/* --------------------------------------------------------------------- */
/*  Error type                                                           */

#[derive(Debug, Error)]
pub enum IoError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("cbor: {0}")]
    Cbor(String),
    #[error("invalid temporary path")]
    TmpPath,
}

type IoResult<T> = Result<T, IoError>;

fn write_cbor<W: Write, T: Serialize + ?Sized>(w: W, val: &T) -> IoResult<()> {
    ser::into_writer(val, w).map_err(|e| IoError::Cbor(e.to_string()))
}
fn read_cbor<R: Read, T: DeserializeOwned>(r: R) -> IoResult<T> {
    de::from_reader(r).map_err(|e| IoError::Cbor(e.to_string()))
}

/* --------------------------------------------------------------------- */
/*  Traits                                                               */

pub trait WakewordSave: Serialize {
    /// Atomically write CBOR to `path`.
    /// Uses “`<file>.tmp` → rename” on the same filesystem for safety.
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> IoResult<()> {
        let path = path.as_ref();
        let tmp = path.with_extension("tmp");

        {
            let f = File::create(&tmp)?;
            let mut bw = BufWriter::new(f);
            write_cbor(&mut bw, self)?;
            bw.flush()?;
        }
        fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Serialize into an in-memory CBOR buffer.
    fn save_to_buffer(&self) -> IoResult<Vec<u8>> {
        let mut buf = Vec::new();
        write_cbor(&mut buf, self)?;
        Ok(buf)
    }
}

pub trait WakewordLoad: DeserializeOwned + Sized {
    /// Load a CBOR file produced by [`WakewordSave::save_to_file`].
    fn load_from_file<P: AsRef<Path>>(path: P) -> IoResult<Self> {
        let f = File::open(path)?;
        read_cbor(BufReader::new(f))
    }

    /// Load from a CBOR buffer (e.g. received over the wire).
    fn load_from_buffer(buf: &[u8]) -> IoResult<Self> {
        read_cbor(BufReader::new(buf))
    }
}

/* --------------------------------------------------------------------- */
/*  Internal trait used by detector code                                 */

#[allow(dead_code)]
pub(crate) trait WakewordFile {
    fn get_detector(
        &self,
        score_ref: f32,
        band_size: u16,
        score_mode: ScoreMode,
    ) -> Box<dyn WakewordDetector>;

    fn get_kfc_size(&self) -> u16;
}
