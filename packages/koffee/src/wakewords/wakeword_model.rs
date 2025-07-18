//---
// path: potter-core/src/wakeword/model.rs
//---

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use strum::{EnumString, IntoStaticStr};

// Custom serialization module for IndexMap
mod indexmap_as_map {
    use super::*;
    use serde::ser::SerializeMap; // trait in scope for serialize_entry/end

    #[allow(dead_code)]
    pub fn serialize<S, K, V>(map: &IndexMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        K: Serialize,
        V: Serialize,
    {
        let mut map_ser = serializer.serialize_map(Some(map.len()))?;
        for (k, v) in map {
            map_ser.serialize_entry(k, v)?;
        }
        serde::ser::SerializeMap::end(map_ser)
    }

    #[allow(dead_code)]
    pub fn deserialize<'de, D, K, V>(deserializer: D) -> Result<IndexMap<K, V>, D::Error>
    where
        D: Deserializer<'de>,
        K: Deserialize<'de> + std::hash::Hash + Eq,
        V: Deserialize<'de>,
    {
        let map = HashMap::<K, V>::deserialize(deserializer)?;
        Ok(map.into_iter().collect())
    }
}

use crate::{
    KoffeeCandleDetection, // new import required by NullDetector implementation
    ScoreMode,
    wakewords::nn::WakewordNN,
    wakewords::{WakewordDetector, WakewordFile, WakewordLoad, WakewordSave},
};

/// Current on-disk version.  Bump if the binary layout changes.
pub const MODEL_VERSION: u8 = 1;

/// Typed wrapper for a CBOR-serialised wake-word neural network.
#[derive(Serialize, Deserialize)]
pub struct WakewordModel {
    /// File format version header.
    version: u8,

    /// Human-readable labels; must contain `"none"`.
    pub labels: Vec<String>,

    /// Number of KFC **frames** expected by the model.
    pub train_size: usize,

    /// Fixed KFC tensor shape expected by the model
    /// (coefficients / frame , frames / sample)
    pub kfc_size: (u16, u16),

    /// Network size preset.
    pub m_type: ModelType,

    /// Raw weight tensors.
    pub weights: ModelWeights,

    /// Median RMS level of the positive samples used in training.
    pub rms_level: f32,
}

impl WakewordModel {
    /// Convenience constructor that validates input.
    #[inline]
    pub fn new(
        labels: Vec<String>,
        train_size: usize,
        kfc_size: (u16, u16),
        m_type: ModelType,
        weights: ModelWeights,
        rms_level: f32,
    ) -> Self {
        Self {
            version: MODEL_VERSION,
            labels,
            train_size,
            kfc_size,
            m_type,
            weights,
            rms_level,
        }
    }
}

/* ----- serde helpers already provided by blanket traits -------------- */

impl WakewordLoad for WakewordModel {}
impl WakewordSave for WakewordModel {}

// Fallback detector that always returns None.
// Kept for completeness, but not used in production path.
#[allow(dead_code)]
struct NullDetector {
    kfc_size: u16,
}

impl NullDetector {
    #[allow(dead_code)]
    fn new(kfc_size: u16) -> Self {
        Self { kfc_size }
    }
}

impl WakewordDetector for NullDetector {
    fn get_kfc_dimensions(&self) -> (u16, usize) {
        (self.kfc_size, 0)
    }

    fn run_detection(
        &self,
        _kfc_frame: Vec<Vec<f32>>,
        _avg_threshold: f32,
        _threshold: f32,
    ) -> Option<KoffeeCandleDetection> {
        None
    }

    fn get_rms_level(&self) -> f32 {
        0.0
    }

    fn update_config(&mut self, _score_ref: f32, _band_size: u16, _score_mode: ScoreMode) {}
}

impl WakewordFile for WakewordModel {
    fn get_detector(
        &self,
        score_ref: f32,
        _band: u16,
        _mode: ScoreMode,
    ) -> Box<dyn WakewordDetector> {
        Box::new(WakewordNN::new(self, score_ref).expect("weight mismatch"))
    }

    // -----------------------------------------------------------------
    // Legacy shim – the lib still asks for "kfc" size in a few places.
    // Keep returning the coeff-count to avoid a breaking change.
    // -----------------------------------------------------------------
    fn get_kfc_size(&self) -> u16 {
        self.kfc_size.0
    }
}

/* --------------------------------------------------------------------- */
/*  ModelType enum                                                       */

#[derive(Clone, Copy, Eq, PartialEq, Debug, Serialize, Deserialize, EnumString, IntoStaticStr)]
#[strum(serialize_all = "lowercase")]
pub enum ModelType {
    Tiny,
    Small,
    Medium,
    Large,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", <&'static str>::from(self))
    }
}

/* --------------------------------------------------------------------- */
/*  Weights map + tensor wrapper                                         */

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorData {
    /// Raw bytes (little-endian f32/f16, etc.)
    pub bytes: Vec<u8>,
    /// Tensor dimensions, row-major.
    pub dims: Vec<usize>,
    /// Candle dtype as lowercase string (`"f32"`, `"bf16"`, …).
    pub d_type: String,
}

/// Model weights - either as a structured map or raw bytes
#[derive(Debug)]
pub enum ModelWeights {
    /// Structured tensor map with deterministic ordering
    Map(IndexMap<String, TensorData>),
    /// Raw bytes (usually from safetensors)
    Raw(Vec<u8>),
}

// Manual implementation of Serialize for ModelWeights
impl Serialize for ModelWeights {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;

        match self {
            ModelWeights::Map(map) => {
                let mut map_ser = serializer.serialize_map(Some(map.len()))?;
                for (k, v) in map {
                    map_ser.serialize_entry(k, v)?;
                }
                map_ser.end()
            }
            ModelWeights::Raw(bytes) => serializer.serialize_bytes(bytes),
        }
    }
}

// Manual implementation of Deserialize for ModelWeights
impl<'de> Deserialize<'de> for ModelWeights {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ModelWeightsVisitor;

        impl<'de> serde::de::Visitor<'de> for ModelWeightsVisitor {
            type Value = ModelWeights;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("either a CBOR map or raw byte blob")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut result = IndexMap::new();
                while let Some((key, value)) = map.next_entry()? {
                    result.insert(key, value);
                }
                Ok(ModelWeights::Map(result))
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(ModelWeights::Raw(v.to_vec()))
            }
        }

        deserializer.deserialize_any(ModelWeightsVisitor)
    }
}

impl From<Vec<u8>> for ModelWeights {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Raw(bytes)
    }
}

impl From<IndexMap<String, TensorData>> for ModelWeights {
    fn from(map: IndexMap<String, TensorData>) -> Self {
        Self::Map(map)
    }
}
