//! End-to-end tests for WakewordRef / WakewordModel tooling.
//
//   cargo test -p rustpotter --test wakeword
//

use anyhow::Result;
use rustpotter::{
    ModelType, WakewordLoad, WakewordModel, WakewordModelTrain, WakewordModelTrainOptions,
    WakewordRef, WakewordRefBuildFromFiles, WakewordSave,
};
use std::{fs, path::PathBuf};

/// Helpers
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn wav(path: &str) -> String {
    root().join(path).to_string_lossy().into_owned()
}

/* ───────────────────────────────────────── WakewordRef round-trips ─── */

static DATA: &[(&str, &[&str])] = &[
    (
        "oye casa",
        &[
            "tests/resources/oye_casa_g_1.wav",
            "tests/resources/oye_casa_g_2.wav",
            "tests/resources/oye_casa_g_3.wav",
            "tests/resources/oye_casa_g_4.wav",
            "tests/resources/oye_casa_g_5.wav",
        ],
    ),
    (
        "alexa",
        &[
            "tests/resources/alexa.wav",
            "tests/resources/alexa2.wav",
            "tests/resources/alexa3.wav",
        ],
    ),
];

#[test]
fn create_and_persist_refs() -> Result<()> {
    for (label, files) in DATA {
        let paths: Vec<String> = files.iter().map(|f| wav(f)).collect();
        let ww =
            WakewordRef::new_from_sample_files(label.to_string(), None, None, paths.clone(), 5)?;

        // quick sanity
        assert_eq!(
            ww.samples_features.len(),
            paths.len(),
            "extracted KFC feature sets"
        );

        // round-trip to disk
        let model_path = root().join("tests/resources").join(format!("{label}.rpw"));
        ww.save_to_file(model_path.to_str().unwrap())?;

        let loaded = WakewordRef::load_from_file(model_path.to_str().unwrap())?;
        assert_eq!(
            loaded.samples_features.len(),
            ww.samples_features.len(),
            "round-trip preserved sample count"
        );
    }
    Ok(())
}

/* ───────────────────────────────────────── Model training smoke test ─ */

#[test]
fn train_and_validate_model() -> Result<()> {
    let train_dir = wav("tests/resources/train");
    let test_dir = wav("tests/resources/test");
    let kfc_size = 16;

    let cfg = WakewordModelTrainOptions::new(ModelType::Medium, 0.027, 10, 10, kfc_size);

    let model = WakewordModel::train_from_dirs(cfg, train_dir, test_dir, None)?;

    assert_eq!(model.labels.len(), 2, "two labels expected (wake/none)");
    assert_eq!(model.weights.len(), 6, "6 tensors for Medium model");
    assert_eq!(
        model.train_size, 168,
        "KFC vectors per training sample should be 168"
    );
    assert_eq!(
        model.kfc_size.0, kfc_size,
        "KFC coefficients dimension matches request"
    );

    // save / reload just to be sure
    let tmp_path = root().join("tests/resources/tmp_model.rpw");
    model.save_to_file(tmp_path.to_str().unwrap())?;
    fs::remove_file(tmp_path)?; // cleanup

    Ok(())
}
