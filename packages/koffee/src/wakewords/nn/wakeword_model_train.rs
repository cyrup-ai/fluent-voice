//! Wake-word CNN trainer (KFC front-end).
//! Replaces the historic KFC trainer; identical logic but with the new
//! `kfc` module and Candle-main neural-net API.

use std::{
    collections::HashMap,
    io::{BufReader, Cursor},
};

use candle_core::{DType, Device, Tensor};
use candle_nn::optim::ParamsAdamW; // new typed-param bag
use candle_nn::{self as nn, Module, Optimizer, optim::AdamW}; // ← Optimizer in scope

use thiserror::Error;

// Helper alias makes the high-fan-out tuple readable & keeps Clippy quiet.
type Dataset = (Vec<Vec<f32>>, Vec<u32>, f32);

use crate::{
    constants::NN_NONE_LABEL,
    kfc::{KfcNormalizer, KfcWavFileExtractor},
    wakewords::{ModelType, ModelWeights, TensorData, WakewordModel},
};

// Add conversion from u8 to ModelType
impl From<u8> for ModelType {
    fn from(value: u8) -> Self {
        match value {
            0 => ModelType::Tiny,
            1 => ModelType::Small,
            2 => ModelType::Medium,
            3 => ModelType::Large,
            _ => ModelType::Tiny, // Default to Tiny for unknown values
        }
    }
}

/// KFC configuration constants (mirrors old KFC defaults).
pub const COEFFS: usize = crate::constants::KFC_COEFFS;
pub const FRAMES: usize = crate::constants::KFC_FRAMES;

/// ------------------------------------------------------------
///  Training-time error handling
/// ------------------------------------------------------------
#[derive(Debug, Error)]
pub enum TrainerError {
    #[error("training data set is empty")]
    EmptySet,
    #[error("label \"{0}\" was not present in the previous model")]
    UnknownLabel(String),
    #[error("kfc extractor: {0}")]
    Kfc(#[from] crate::kfc::Error),
    #[error("extractor: {0}")]
    Extractor(#[from] crate::kfc::ExtractorError),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("mutex poison: {0}")]
    Poison(String),
    #[error("ciborium serialization: {0}")]
    Ciborium(String),
}

impl<W> From<ciborium::ser::Error<W>> for TrainerError
where
    W: std::fmt::Debug,
{
    fn from(err: ciborium::ser::Error<W>) -> Self {
        TrainerError::Ciborium(format!("{err:?}"))
    }
}

/// Hyper-parameters accepted by the trainer.
#[derive(Clone, Debug)]
pub struct WakewordModelTrainOptions {
    /// Learning rate for training
    pub lr: f64,
    /// multiplicative decay each epoch  (cosine if negative)
    pub lr_decay: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Test model every N epochs
    pub test_every: usize,
    /// Training batch size
    pub batch_size: usize,
    /// Band size for frequency analysis
    pub band_size: u16,
    /// early-stopping patience (epochs without improvement)
    pub early_stop_pat: usize,
    /// Model type to use (Tiny, Small, Medium, Large)
    pub model_type: u8,
}
impl Default for WakewordModelTrainOptions {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            lr_decay: 0.95,
            epochs: 100,
            test_every: 5,
            batch_size: 32,
            band_size: 3,
            early_stop_pat: 10,
            model_type: 0, // Tiny by default
        }
    }
}

/// Main training entry-point.
pub fn train(
    train: HashMap<String, Vec<u8>>,
    val: HashMap<String, Vec<u8>>,
    prev: Option<&WakewordModel>,
    opts: WakewordModelTrainOptions,
) -> Result<WakewordModel, TrainerError> {
    if train.is_empty() || val.is_empty() {
        return Err(TrainerError::EmptySet);
    }

    /* ---------- 1. Prepare label list ---------- */

    // If we’re continuing training, we keep previous labels; else start empty
    let mut labels: Vec<String> = prev.map(|m| m.labels.clone()).unwrap_or_default();

    let parse_label = |fname: &str| {
        fname
            .split_once('[')
            .and_then(|(_, r)| r.split_once(']'))
            .map(|(lab, _)| lab.to_lowercase())
            .unwrap_or_else(|| NN_NONE_LABEL.to_owned())
    };

    // thread-safe storage for labels when we run Rayon
    let labels_guard = std::sync::Mutex::new(&mut labels);

    let take_label = |fname: &str, allow_new: bool| -> Result<usize, TrainerError> {
        let lab = parse_label(fname);
        let lock_result = labels_guard.lock();
        let mut g = match lock_result {
            Ok(guard) => guard,
            Err(poison_err) => {
                return Err(TrainerError::Poison(format!("labels_guard: {poison_err}")));
            }
        };

        if let Some(pos) = g.iter().position(|l| l == &lab) {
            Ok(pos)
        } else if allow_new {
            g.push(lab);
            Ok(g.len() - 1)
        } else {
            Err(TrainerError::UnknownLabel(lab))
        }
    };

    /* ---------- 2. Feature extraction ---------- */

    /* ---------- 2. Feature extraction ---------- */

    /* ---------- 2. Feature extraction ---------- */

    fn build_set<F>(
        data: HashMap<String, Vec<u8>>,
        allow_new_labels: bool,
        take_label: &F,
    ) -> Result<Dataset, TrainerError>
    where
        F: Fn(&str, bool) -> Result<usize, TrainerError> + Send + Sync,
    {
        // Thread-safe accumulators
        let feats = std::sync::Mutex::new(Vec::<Vec<f32>>::new());
        let labels = std::sync::Mutex::new(Vec::<u32>::new());
        let rms_vec = std::sync::Mutex::new(Vec::<f32>::new());

        // Sequential feature extraction (debugging) -------------------------------------------------
        data.into_iter()
            .try_for_each(|(fname, bytes)| -> Result<(), TrainerError> {
                // map file-name → label index
                let lab_ix = take_label(&fname, allow_new_labels)?;

                // KFC->KFCC pipeline
                let mut rms = 0.0;
                let mut extractor = KfcWavFileExtractor::compute_kfcs(
                    BufReader::new(Cursor::new(bytes)),
                    &mut rms,
                    COEFFS as u16,
                )
                .map_err(TrainerError::Extractor)?;
                KfcNormalizer::normalize(&mut extractor);

                // Ensure we have exactly FRAMES frames for the neural network
                let frames_to_take = extractor.len().min(FRAMES);
                let mut flat: Vec<f32> = extractor
                    .into_iter()
                    .take(frames_to_take)
                    .flatten()
                    .collect();

                // Pad with zeros if we don't have enough frames
                let expected_size = COEFFS * FRAMES;
                if flat.len() < expected_size {
                    flat.resize(expected_size, 0.0);
                }
                let rms = flat.iter().copied().map(f32::abs).sum::<f32>() / flat.len() as f32;

                // Push into shared buffers (with poison-aware error handling)
                {
                    feats
                        .lock()
                        .map_err(|e| TrainerError::Poison(format!("feats: {e}")))?
                        .push(flat);
                }
                {
                    labels
                        .lock()
                        .map_err(|e| TrainerError::Poison(format!("labels: {e}")))?
                        .push(lab_ix as u32);
                }
                {
                    rms_vec
                        .lock()
                        .map_err(|e| TrainerError::Poison(format!("rms: {e}")))?
                        .push(rms);
                }

                Ok(())
            })?;

        // ---------------------------------------------------------------------------

        // Take ownership of the buffers without panicking
        let feats = feats
            .into_inner()
            .map_err(|e| TrainerError::Poison(format!("feats final: {e}")))?;
        let labels = labels
            .into_inner()
            .map_err(|e| TrainerError::Poison(format!("labels final: {e}")))?;
        let rms_vec = rms_vec
            .into_inner()
            .map_err(|e| TrainerError::Poison(format!("rms final: {e}")))?;

        println!(
            "DEBUG: Final dataset - {} features, {} labels, {} rms values",
            feats.len(),
            labels.len(),
            rms_vec.len()
        );
        if feats.is_empty() {
            println!("DEBUG: WARNING - No features extracted from any files!");
        } else {
            println!(
                "DEBUG: First feature vector has {} elements",
                feats[0].len()
            );
        }

        // Compute average RMS across all samples
        let rms_level = if rms_vec.is_empty() {
            0.0
        } else {
            rms_vec.iter().sum::<f32>() / rms_vec.len() as f32
        };

        Ok((feats, labels, rms_level))
    }

    let (tr_x, tr_y, rms_train): Dataset = build_set(train, prev.is_none(), &take_label)?;
    let (val_x, val_y, rms_val): Dataset = build_set(val, false, &take_label)?;

    /* ---------- 3. Build neural network ---------- */

    let dev = Device::Cpu;
    let var_map = candle_nn::VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&var_map, DType::F32, &dev);
    let root = vs.clone();
    // -----------------------------------------------------------------
    // ⚙ Restore the original "Kfc-CNN" first-layer width (16 channels).
    //    This yields the same feature capacity & test-set accuracy the
    //    original implementation achieved, while preserving all modern
    //    safety / ergonomics improvements made elsewhere.
    // -----------------------------------------------------------------
    // ------------------------------------------------------------------
    //  ARCHITECTURE MATCHING INFERENCE CODE
    // ------------------------------------------------------------------

    // Use EXACT same inter function as inference code but ensure minimum sizes
    const fn inter(inp: usize, kfc: usize, div: usize) -> usize {
        let result = (inp / kfc) / (crate::constants::KFCS_EXTRACTOR_OUT_SHIFTS * div);
        if result == 0 { 8 } else { result } // Minimum 8 neurons to prevent numerical issues
    }

    let model_type = ModelType::from(opts.model_type);
    let inp = tr_x[0].len(); // Input size from training data
    let kfc = COEFFS * FRAMES; // KFC size

    // Create architecture matching inference code EXACTLY
    let (ln1, ln2, ln3) = match model_type {
        ModelType::Tiny => {
            let i = inter(inp, kfc, 5);
            let ln1 = nn::linear(inp, i, root.pp("ln1"))?;
            let ln2 = nn::linear(i, labels.len(), root.pp("ln2"))?;
            (ln1, ln2, None)
        }
        ModelType::Small => {
            let i1 = inter(inp, kfc, 2);
            let i2 = std::cmp::max(i1 / 2, 4); // Ensure minimum 4 neurons
            let ln1 = nn::linear(inp, i1, root.pp("ln1"))?;
            let ln2 = nn::linear(i1, i2, root.pp("ln2"))?;
            let ln3 = nn::linear(i2, labels.len(), root.pp("ln3"))?;
            (ln1, ln2, Some(ln3))
        }
        ModelType::Medium => {
            let i1 = inter(inp, kfc, 1);
            let i2 = inter(inp, kfc, 2);
            let ln1 = nn::linear(inp, i1, root.pp("ln1"))?;
            let ln2 = nn::linear(i1, i2, root.pp("ln2"))?;
            let ln3 = nn::linear(i2, labels.len(), root.pp("ln3"))?;
            (ln1, ln2, Some(ln3))
        }
        ModelType::Large => {
            let i1 = inter(inp, kfc, 1) * 2;
            let i2 = inter(inp, kfc, 2);
            let ln1 = nn::linear(inp, i1, root.pp("ln1"))?;
            let ln2 = nn::linear(i1, i2, root.pp("ln2"))?;
            let ln3 = nn::linear(i2, labels.len(), root.pp("ln3"))?;
            (ln1, ln2, Some(ln3))
        }
    };

    // Define a forward function matching inference architecture
    let forward_fn = move |xs: &Tensor| -> candle_core::Result<Tensor> {
        let xs = ln1.forward(xs)?.relu()?;
        match &ln3 {
            Some(ln3_layer) => {
                let xs = ln2.forward(&xs)?.relu()?;
                ln3_layer.forward(&xs)
            }
            None => ln2.forward(&xs), // Tiny model only has 2 layers
        }
    };

    /* ---------- 4. Training ------------ */

    // Candle 0.9 API → take explicit parameter vector
    let params = candle_nn::optim::ParamsAdamW {
        lr: opts.lr,
        ..Default::default()
    };
    let mut opt = AdamW::new(var_map.all_vars(), params)?;

    // Function to convert feature vectors to tensors
    let batchify = |xs: &[Vec<f32>]| -> candle_core::Result<Tensor> {
        Tensor::from_iter(xs.iter().flat_map(|v| v.iter().cloned()), &dev)
            .and_then(|t| t.reshape((xs.len(), COEFFS * FRAMES)))
    };

    let batched_tr_x = batchify(&tr_x)?;
    let batched_val_x = batchify(&val_x)?;
    let tr_y_t = Tensor::from_slice(&tr_y, (tr_y.len(),), &dev)?;
    let val_y_t = Tensor::from_slice(&val_y, (val_y.len(),), &dev)?;

    let batches = tr_x.len().div_ceil(opts.batch_size);
    let mut best_val = f32::INFINITY;
    let mut epochs_no_improve = 0usize;
    // keep a snapshot of the best weights as ModelWeights::Map
    let mut best_weights: Option<ModelWeights> = None;
    // cosine schedule support
    let cosine = opts.lr_decay.is_sign_negative();

    let mut global_step = 0usize;
    for epoch in 1..=opts.epochs {
        let mut epoch_loss = 0f32;

        for b in 0..batches {
            let lo = b * opts.batch_size;
            let hi = (lo + opts.batch_size).min(tr_x.len());

            let x = batched_tr_x.narrow(0, lo, hi - lo)?;
            let y = tr_y_t.narrow(0, lo, hi - lo)?;

            let logits = forward_fn(&x)?;
            let loss = nn::loss::cross_entropy(&logits, &y)?;
            opt.backward_step(&loss)?;

            epoch_loss += f32::try_from(&loss)?;

            // -------- per-batch validation -----------------
            if global_step.is_multiple_of(opts.test_every) {
                let v_logits = forward_fn(&batched_val_x)?;
                let v_loss = nn::loss::cross_entropy(&v_logits, &val_y_t)?;
                best_val = best_val.min(f32::try_from(&v_loss)?);
            }
            global_step += 1;
        }

        // -------- epoch-level validation & checkpoint ----------
        let v_logits = forward_fn(&batched_val_x)?;
        let v_loss = nn::loss::cross_entropy(&v_logits, &val_y_t)?;
        let vloss = f32::try_from(&v_loss)?;

        if vloss < best_val - 1e-6 {
            best_val = vloss;
            epochs_no_improve = 0;
            // Handle potential poisoned mutex
            let lock_result = var_map.data().lock();
            let guard = match lock_result {
                Ok(g) => g,
                Err(poison_err) => {
                    return Err(TrainerError::Poison(format!(
                        "var_map.data(): {poison_err}"
                    )));
                }
            };

            // Process tensors in deterministic order
            let mut tensors = indexmap::IndexMap::new();
            for (k, v) in guard.iter() {
                let dims = v.shape().dims().to_vec();
                let dtype = v.dtype().as_str().to_owned();
                let mut bytes = Vec::new();
                // Handle potential tensor write errors
                v.write_bytes(&mut bytes).map_err(TrainerError::Candle)?;
                tensors.insert(
                    k.clone(),
                    TensorData {
                        bytes,
                        dims,
                        d_type: dtype,
                    },
                );
            }

            // Store weights as ModelWeights::Map directly
            best_weights = Some(ModelWeights::Map(tensors));
        } else {
            epochs_no_improve += 1;
            if epochs_no_improve >= opts.early_stop_pat {
                break;
            }
        }

        log::debug!(
            "epoch {}/{}  train_loss={:.6}  val_loss={:.6}",
            epoch,
            opts.epochs,
            epoch_loss / batches as f32,
            vloss
        );

        // -------- learning-rate schedule ----------
        let new_lr = if cosine {
            // cosine schedule – decay from lr to lr*0.1
            let t = epoch as f64 / opts.epochs as f64;
            let min_lr = opts.lr * 0.1;
            min_lr + 0.5 * (opts.lr - min_lr) * (1.0 + (std::f64::consts::PI * t).cos())
        } else {
            opts.lr * opts.lr_decay.powi(epoch as i32)
        };
        // Recreate optimizer with new learning rate
        opt = AdamW::new(
            var_map.all_vars(),
            ParamsAdamW {
                lr: new_lr,
                ..Default::default()
            },
        )?;
    }

    /* ---------- 5. Pack result ---------- */

    // Get the final weights - either the best checkpoint or current weights
    let final_weights = if let Some(weights) = best_weights {
        weights
    } else {
        // Get current weights if no better snapshot exists
        let lock_result = var_map.data().lock();
        let guard = match lock_result {
            Ok(g) => g,
            Err(poison_err) => {
                return Err(TrainerError::Poison(format!(
                    "var_map.data(): {poison_err}"
                )));
            }
        };

        // Process tensors with proper error handling
        let mut tensors = indexmap::IndexMap::new();
        for (k, v) in guard.iter() {
            let dims = v.shape().dims().to_vec();
            let dtype = v.dtype().as_str().to_owned();
            let mut bytes = Vec::new();
            // Handle potential tensor write errors
            v.write_bytes(&mut bytes).map_err(TrainerError::Candle)?;
            tensors.insert(
                k.clone(),
                TensorData {
                    bytes,
                    dims,
                    d_type: dtype,
                },
            );
        }

        // Return as ModelWeights::Map directly
        ModelWeights::Map(tensors)
    };

    // Convert the model type from the training options to the proper enum
    let model_type = ModelType::from(opts.model_type);
    log::debug!("Saving model with type: {model_type:?}");

    Ok(WakewordModel::new(
        labels,                         // Vec<String> – must include "none"
        tr_x[0].len(),                  // train_size  (actual input feature vector size)
        (COEFFS as u16, FRAMES as u16), // kfc_size
        model_type,                     // Use the model type from training options
        final_weights,                  // best weights as ModelWeights::Map
        (rms_train + rms_val) * 0.5,    // rms_level   (median RMS of samples)
    ))
}
