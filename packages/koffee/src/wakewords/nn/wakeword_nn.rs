#![allow(clippy::too_many_arguments)]

use std::{collections::HashMap, io::Cursor, sync::Mutex};

use candle_core::Module; // <-- brings `forward()` into scope
use candle_core::{DType, Device, Result as CandleResult, Tensor, Var};
use candle_nn::{Linear, VarBuilder, VarMap};
use indexmap::IndexMap;

use crate::{
    KoffeeCandleDetection, ModelType, ScoreMode, WakewordModel,
    constants::{KFCS_EXTRACTOR_OUT_SHIFTS, NN_NONE_LABEL},
    kfc::KfcNormalizer,
    wakewords::{ModelWeights, TensorData, WakewordDetector},
};

/* ------------------------------------------------------------------------- */
/*  Error handling                                                           */
/* ------------------------------------------------------------------------- */

#[derive(Debug, thiserror::Error)]
pub enum WakewordError {
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("tensor-io: {0}")]
    Io(#[from] std::io::Error),
    #[error("model weight '{0}' missing in checkpoint")]
    MissingWeight(String),
    #[error("tensor data malformed: {0}")]
    TensorData(String),
}

/* ------------------------------------------------------------------------- */
/*  Public struct                                                            */
/* ------------------------------------------------------------------------- */

pub struct WakewordNN {
    model: Box<dyn ModelImpl>,
    // tensor reused between calls (avoids realloc); wrapped in `Mutex` to
    // provide thread safety for concurrent access
    scratch: Mutex<Tensor>,
    kfc_frames: usize,
    kfc_coeffs: u16,
    labels: Vec<String>,
    score_scale: f32, // = score_ref * 10
    #[allow(dead_code)]
    rms_level: f32,
}

impl WakewordNN {
    /* ---------------- constructor ---------------- */

    pub fn new(model: &WakewordModel, score_ref: f32) -> Result<Self, WakewordError> {
        let var_map = VarMap::new();
        let model_size = model.train_size; // train_size now contains the actual input feature vector size

        let net = init_model(
            model.m_type,
            &var_map,
            &Device::Cpu,
            model_size,
            model.kfc_size.0 * model.kfc_size.1, // Use COEFFS * FRAMES like training
            model.labels.len(),
            Some(model),
        )?;

        let scratch = Mutex::new(Tensor::zeros((1, model_size), DType::F32, &Device::Cpu)?);

        Ok(Self {
            model: net,
            scratch,
            kfc_frames: model.train_size,
            kfc_coeffs: model.kfc_size.0,
            labels: model.labels.clone(),
            score_scale: score_ref * 10.0,
            rms_level: model.rms_level,
        })
    }

    /* ---------------- helpers -------------------- */

    #[inline]
    fn top_label<'a>(&'a self, probs: &'a [f32]) -> (&'a str, f32, f32) {
        let (mut best_idx, mut best_p) = (usize::MAX, f32::MIN);
        let mut none_p = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best_idx = i;
            }
            if i < self.labels.len() && self.labels[i] == NN_NONE_LABEL {
                none_p = p;
            }
        }

        let lbl = self
            .labels
            .get(best_idx)
            .map(|s| s.as_str())
            .unwrap_or(NN_NONE_LABEL);

        (lbl, best_p, none_p)
    }

    #[inline]
    fn inv_similarity(&self, p1: f32, p2: f32) -> f32 {
        1.0 - (1.0 / (1.0 + ((p1 - p2 - self.score_scale) / self.score_scale).exp()))
    }

    fn flatten_frames(&self, mut kfc: Vec<Vec<f32>>) -> Vec<f32> {
        kfc.truncate(self.kfc_frames);
        KfcNormalizer::normalize(&mut kfc);

        let needed = self.kfc_frames * self.kfc_coeffs as usize;
        let mut flat = Vec::with_capacity(needed);

        for frame in kfc {
            flat.extend(frame);
        }
        flat.resize(needed, 0.0);
        flat
    }

    fn predict_internal(
        &self,
        flat: Vec<f32>,
        scratch: &mut Tensor,
    ) -> Result<Vec<f32>, WakewordError> {
        // Create tensor with correct input dimensions: (batch_size=1, input_features)
        let input_features = flat.len();
        *scratch = Tensor::from_vec(flat, (1, input_features), &Device::Cpu)?;
        let logits = self.model.forward(scratch)?;
        Ok(logits.get(0)?.to_vec1::<f32>()?)
    }

    /* ------------- public prediction API ---------- */

    #[allow(dead_code)]
    fn predict(&mut self, kfc: Vec<Vec<f32>>) -> Result<Vec<f32>, WakewordError> {
        let flat = self.flatten_frames(kfc);
        let mut scratch_ref = self
            .scratch
            .lock()
            .map_err(|e| WakewordError::TensorData(format!("Mutex lock failed: {}", e)))?;
        self.predict_internal(flat, &mut scratch_ref)
    }

    fn predict_with_scratch(
        &self,
        kfc: Vec<Vec<f32>>,
        scratch: &mut Tensor,
    ) -> Result<Vec<f32>, WakewordError> {
        let flat = self.flatten_frames(kfc);
        self.predict_internal(flat, scratch)
    }
}

/* ------------------------------------------------------------------------- */
/*  WakewordDetector impl                                                    */
/* ------------------------------------------------------------------------- */

impl WakewordDetector for WakewordNN {
    fn run_detection(
        &self,
        kfc_frames: Vec<Vec<f32>>,
        avg_th: f32,
        th: f32,
    ) -> Option<KoffeeCandleDetection> {
        // lock the mutex scratch without aliasing `self`
        let mut local = self.scratch.lock().ok()?;
        let probs = self.predict_with_scratch(kfc_frames, &mut local).ok()?;
        let (lbl, p_top, p_none) = self.top_label(&probs);

        if lbl == NN_NONE_LABEL {
            return None;
        }

        let avg_score = if avg_th > 0.0 {
            let p_second = probs
                .iter()
                .copied()
                .filter(|&p| (p - p_top).abs() > f32::EPSILON)
                .fold(f32::MIN, f32::max);

            self.inv_similarity(p_top, p_second)
        } else {
            0.0
        };

        let score = self.inv_similarity(p_top, p_none);

        if score < th || avg_score < avg_th {
            return None;
        }

        let scores: HashMap<_, _> = self
            .labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), probs[i]))
            .collect();

        Some(KoffeeCandleDetection {
            name: lbl.to_string(),
            avg_score,
            score,
            scores,
            counter: usize::MIN,
            gain: f32::NAN,
        })
    }

    fn get_rms_level(&self) -> f32 {
        self.rms_level
    }

    fn get_kfc_dimensions(&self) -> (u16, usize) {
        (self.kfc_coeffs, self.kfc_frames)
    }

    fn update_config(&mut self, score_ref: f32, _: u16, _: ScoreMode) {
        self.score_scale = score_ref * 10.0;
    }
}

/* ------------------------------------------------------------------------- */
/*  Utilities                                                                */
/* ------------------------------------------------------------------------- */

#[allow(dead_code)]
pub(super) fn get_tensors_data(vm: &VarMap) -> Result<ModelWeights, WakewordError> {
    let mut map = IndexMap::new();
    for (name, var) in vm
        .data()
        .lock()
        .map_err(|_| WakewordError::TensorData("mutex".into()))?
        .iter()
    {
        map.insert(name.clone(), TensorData::try_from(var)?);
    }
    Ok(ModelWeights::Map(map))
}

/* ------------------------------------------------------------------------- */
/*  Model factory                                                            */
/* ------------------------------------------------------------------------- */

fn init_model(
    m_type: ModelType,
    var_map: &VarMap,
    dev: &Device,
    feat_size: usize,
    kfc_coeffs: u16,
    n_labels: usize,
    ckpt: Option<&WakewordModel>,
) -> Result<Box<dyn ModelImpl>, WakewordError> {
    let boxed: Box<dyn ModelImpl> = match m_type {
        ModelType::Tiny => {
            init_model_impl::<TinyModel>(var_map, dev, feat_size, kfc_coeffs, n_labels, ckpt)?
        }
        ModelType::Small => {
            init_model_impl::<SmallModel>(var_map, dev, feat_size, kfc_coeffs, n_labels, ckpt)?
        }
        ModelType::Medium => {
            init_model_impl::<MediumModel>(var_map, dev, feat_size, kfc_coeffs, n_labels, ckpt)?
        }
        ModelType::Large => {
            init_model_impl::<LargeModel>(var_map, dev, feat_size, kfc_coeffs, n_labels, ckpt)?
        }
    };
    Ok(boxed)
}

fn init_model_impl<M: ModelImpl + 'static>(
    vm: &VarMap,
    dev: &Device,
    feat: usize,
    kfc: u16,
    labels: usize,
    ckpt: Option<&WakewordModel>,
) -> Result<Box<dyn ModelImpl>, WakewordError> {
    let vs = VarBuilder::from_varmap(vm, DType::F32, dev);
    let net = M::new(vs.clone(), feat, kfc as usize, labels)?;

    if let Some(old) = ckpt {
        try_load_weights(vm, &old.weights)?;
    }
    Ok(Box::new(net))
}

fn try_load_weights(vm: &VarMap, ckpt: &ModelWeights) -> Result<(), WakewordError> {
    let map = match ckpt {
        ModelWeights::Map(m) => m,
        ModelWeights::Raw(_) => {
            return Err(WakewordError::TensorData(
                "raw safetensor weights not yet supported in loader".into(),
            ));
        }
    };

    for (name, var) in vm
        .data()
        .lock()
        .map_err(|_| WakewordError::TensorData("mutex".into()))?
        .iter_mut()
    {
        match map.get(name) {
            Some(td) => {
                let t = Tensor::try_from(td)?;
                var.set(&t)?;
            }
            None => return Err(WakewordError::MissingWeight(name.clone())),
        }
    }
    Ok(())
}

/* ------------------------------------------------------------------------- */
/*  Linear-stack architectures                                               */
/* ------------------------------------------------------------------------- */

pub(super) struct TinyModel {
    ln1: Linear,
    ln2: Linear,
}

pub(super) struct SmallModel {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}

pub(super) struct MediumModel {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}

pub(super) struct LargeModel {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}

const fn inter(inp: usize, kfc: usize, div: usize) -> usize {
    let result = (inp / kfc) / (KFCS_EXTRACTOR_OUT_SHIFTS * div);
    if result == 0 { 8 } else { result } // Minimum 8 neurons to prevent numerical issues
}

macro_rules! impl_model {
    ($ty:ident, $layers:expr) => {
        impl ModelImpl for $ty {
            fn new(vs: VarBuilder, inp: usize, kfc: usize, labels: usize) -> CandleResult<Self> {
                let (i1, i2) = $layers(inp, kfc);
                let ln1 = candle_nn::linear(inp, i1, vs.pp("ln1"))?;
                let ln2 = candle_nn::linear(i1, i2, vs.pp("ln2"))?;
                let ln3 = candle_nn::linear(i2, labels, vs.pp("ln3"))?;
                Ok(Self { ln1, ln2, ln3 })
            }

            fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
                let xs = self.ln1.forward(xs)?.relu()?;
                let xs = self.ln2.forward(&xs)?.relu()?;
                self.ln3.forward(&xs)
            }
        }
    };
}

/* -- architectures ------------------------------------------------------ */

impl TinyModel {
    fn new(vs: VarBuilder, inp: usize, kfc: usize, labels: usize) -> CandleResult<Self> {
        let i = inter(inp, kfc, 5);
        Ok(Self {
            ln1: candle_nn::linear(inp, i, vs.pp("ln1"))?,
            ln2: candle_nn::linear(i, labels, vs.pp("ln2"))?,
        })
    }
}

impl ModelImpl for TinyModel {
    fn new(vs: VarBuilder, inp: usize, kfc: usize, labels: usize) -> CandleResult<Self> {
        TinyModel::new(vs, inp, kfc, labels)
    }
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let xs = self.ln1.forward(xs)?.relu()?;
        self.ln2.forward(&xs)
    }
}

impl_model!(SmallModel, |i, k| {
    let i1 = inter(i, k, 2);
    let i2 = std::cmp::max(i1 / 2, 4); // Ensure minimum 4 neurons
    (i1, i2)
});
impl_model!(MediumModel, |i, k| {
    let i1 = inter(i, k, 1);
    let i2 = inter(i, k, 2);
    (i1, i2)
});
impl_model!(LargeModel, |i, k| {
    let i1 = inter(i, k, 1) * 2;
    let i2 = inter(i, k, 2);
    (i1, i2)
});

/* ------------------------------------------------------------------------- */
/*  Traits & conversions                                                    */
/* ------------------------------------------------------------------------- */

pub trait ModelImpl: Send + Sync {
    fn new(vs: VarBuilder, inp: usize, kfc: usize, labels: usize) -> CandleResult<Self>
    where
        Self: Sized;
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor>;
}

/* ---------------- TensorData conversions -------------------------------- */

impl TryFrom<&Var> for TensorData {
    type Error = WakewordError;
    fn try_from(var: &Var) -> Result<Self, Self::Error> {
        let mut buf = Cursor::new(Vec::new());
        var.write_bytes(&mut buf)?;
        Ok(TensorData {
            bytes: buf.into_inner(),
            dims: var.shape().dims().to_vec(),
            d_type: var.dtype().as_str().to_owned(),
        })
    }
}

impl TryFrom<&TensorData> for Tensor {
    type Error = WakewordError;

    fn try_from(td: &TensorData) -> Result<Self, Self::Error> {
        let dt = match td.d_type.as_str() {
            "f32" => DType::F32,
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            "i64" => DType::I64,
            "i32" | "i16" | "i8" => DType::I64,
            "u32" | "u8" => DType::U32,
            other => {
                return Err(WakewordError::TensorData(format!(
                    "unsupported dtype {other}"
                )));
            }
        };
        Tensor::from_raw_buffer(&td.bytes, dt, &td.dims, &Device::Cpu).map_err(WakewordError::from)
    }
}
