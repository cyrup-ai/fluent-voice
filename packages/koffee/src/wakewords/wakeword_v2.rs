use std::collections::HashMap;

use crate::{WakewordLoad, wakewords::WakewordRef};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/* --------------------------------------------------------------------- */
/*  Error type                                                            */

#[derive(Debug, Error)]
pub enum V2Error {
    #[error("wakeword contains no templates")]
    Empty,
    #[error("templates have different KFC sizes")]
    InconsistentSize,
}

/* --------------------------------------------------------------------- */
/*  RefError - needed for reference by the TryFrom impl                  */

#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum RefError {
    #[error("wakeword contains no templates")]
    Empty,
    #[error("templates have different KFC sizes")]
    MismatchedSize,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/* --------------------------------------------------------------------- */
/*  Legacy struct (unchanged)                                             */

/// **Deprecated** – on-disk representation used by rustpotter v2.
#[derive(Serialize, Deserialize)]
pub struct WakewordV2 {
    pub name: String,
    pub avg_features: Option<Vec<Vec<f32>>>,
    pub samples_features: HashMap<String, Vec<Vec<f32>>>,
    pub threshold: Option<f32>,
    pub avg_threshold: Option<f32>,
    pub rms_level: f32,
    pub enabled: bool,
}

impl WakewordLoad for WakewordV2 {}

/* --------------------------------------------------------------------- */
/*  Conversion to new format                                              */

impl TryFrom<WakewordV2> for WakewordRef {
    type Error = V2Error;

    fn try_from(v2: WakewordV2) -> Result<Self, Self::Error> {
        if v2.samples_features.is_empty() {
            return Err(V2Error::Empty);
        }
        let kfc_size = v2.samples_features.values().next().ok_or(V2Error::Empty)?[0].len() as u16;

        // Validate all samples share the same KFC dimensionality
        if v2
            .samples_features
            .values()
            .any(|t| t[0].len() as u16 != kfc_size)
        {
            return Err(V2Error::InconsistentSize);
        }

        // Normalise order for determinism
        let samples: IndexMap<_, _> = v2
            .samples_features
            .into_iter() // v stays Vec<Vec<f32>>
            .collect();

        // Match the parameter order in WakewordRef::new
        WakewordRef::new(
            v2.name,
            v2.threshold,
            v2.avg_threshold,
            v2.avg_features,
            v2.rms_level,
            samples.into_iter().collect(),
        )
        // Convert the String error to V2Error
        .map_err(|_| V2Error::InconsistentSize)
    }
}
