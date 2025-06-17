//! If `rustpotter-cli` isn't installed, grab it once during build.
//! Then ensure `assets/wake-word.rpw` exists (otherwise create an empty stub
//! so the project compiles even before training).

use std::{path::Path, process::Command};

fn main() {
    // 1. Ensure the helper CLI is on $PATH (installs once per toolchain dir).
    if Command::new("rustpotter-cli")
        .arg("--version")
        .output()
        .is_err()
    {
        println!("cargo:warning=Installing rustpotter-cli …");
        let status = Command::new("cargo")
            .args(["install", "rustpotter-cli", "--locked"])
            .status()
            .expect("failed to spawn cargo install rustpotter-cli");
        assert!(status.success(), "couldn't install rustpotter-cli");
    }

    // 2. Ensure the model file exists so examples/tests build without training.
    let out = Path::new("assets").join("wake-word.rpw");
    if !out.exists() {
        std::fs::create_dir_all("assets").unwrap();
        std::fs::write(&out, []).unwrap(); // zero-byte placeholder
        println!(
            "cargo:warning=Created stub assets/wake-word.rpw – run `cargo run --bin train-wake-word` to train"
        );
    }

    // Re-run if this file changes.
    println!("cargo:rerun-if-changed=build.rs");
}
