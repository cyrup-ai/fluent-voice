use super::{WakewordDetector, WakewordFile, comp::WakewordComparator};
use crate::{
    ScoreMode,
    kfc::KfcComparator,
    wakewords::{WakewordLoad, WakewordSave},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum RefError {
    #[error("wakeword contains no templates")]
    Empty,
    #[error("templates have different KFC sizes")]
    MismatchedSize,
}

/// Wakeword representation.
#[derive(Serialize, Deserialize)]
pub struct WakewordRef {
    pub name: String,
    pub avg_features: Option<Vec<Vec<f32>>>,
    pub samples_features: HashMap<String, Vec<Vec<f32>>>,
    pub threshold: Option<f32>,
    pub avg_threshold: Option<f32>,
    pub rms_level: f32,
    pub kfc_size: u16,
}
impl WakewordLoad for WakewordRef {}
impl WakewordSave for WakewordRef {}
impl WakewordFile for WakewordRef {
    fn get_detector(
        &self,
        score_ref: f32,
        band_size: u16,
        score_mode: ScoreMode,
    ) -> Box<dyn WakewordDetector> {
        Box::new(WakewordComparator::new(
            self,
            KfcComparator::new(score_ref, band_size),
            score_mode,
        ))
    }

    fn get_kfc_size(&self) -> u16 {
        self.kfc_size
    }
}
impl WakewordRef {
    pub(crate) fn new(
        name: String,
        threshold: Option<f32>,
        avg_threshold: Option<f32>,
        avg_features: Option<Vec<Vec<f32>>>,
        rms_level: f32,
        samples_features: HashMap<String, Vec<Vec<f32>>>,
    ) -> Result<WakewordRef, String> {
        if samples_features.is_empty() {
            return Err("Can not create an empty wakeword".to_string());
        }

        // Get the first sample's features or return an error if none exist
        let first_sample = samples_features
            .values()
            .next()
            .ok_or_else(|| "No samples available in features map".to_string())?;

        // Check that the first sample has at least one feature vector
        if first_sample.is_empty() {
            return Err("First sample contains no feature vectors".to_string());
        }

        let kfc_size = first_sample[0].len() as u16;
        Ok(WakewordRef {
            name,
            threshold,
            avg_threshold,
            avg_features,
            samples_features,
            rms_level,
            kfc_size,
        })
    }
}
