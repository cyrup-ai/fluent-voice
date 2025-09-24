//! If `rustpotter-cli` isn't installed, grab it once during build.
//! Then ensure `assets/wake-word.rpw` exists (otherwise create an empty stub
//! so the project compiles even before training).

use std::{path::Path, process::Command};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Ensure the helper CLI is on $PATH (installs once per toolchain dir).
    if Command::new("rustpotter-cli")
        .arg("--version")
        .output()
        .is_err()
    {
        println!("cargo:warning=Installing rustpotter-cli …");
        let status = Command::new("cargo")
            .args(["install", "rustpotter-cli", "--locked"])
            .status()?; // Remove unwrap
        assert!(status.success(), "couldn't install rustpotter-cli");
    }

    // 2. Check for model file and fail if missing
    let out = Path::new("assets").join("wake-word.rpw");
    if !out.exists() {
        std::fs::create_dir_all("assets")?;

        eprintln!("❌ Wake word model not found: {}", out.display());
        eprintln!("📋 To fix this:");
        eprintln!("   1. Copy a trained .rpw model to assets/wake-word.rpw");
        eprintln!("   2. Or train a new model using: cargo run --example train_syrup");
        eprintln!("   3. Or use an existing model from: ../koffee/training/models/");

        return Err("Required wake word model file missing".into());
    }

    // Re-run if this file changes.
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
