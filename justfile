# ────────────────────────────── Workspace Commands ──────────────────────────────
default:
    @just --list

# Make all Cargo.toml files read-only across the workspace
lock:
    rg --files --type toml | rg '^.*Cargo\.toml$' | xargs chmod 444
    chmod 444 justfile README.md packages/fluent-voice/examples/stt.rs packages/fluent-voice/examples/tts.rs
    @echo "All Cargo.toml files and protected files are now read-only"

# Make all Cargo.toml files writable again
unlock:
    rg --files --type toml | rg '^.*Cargo\.toml$' | xargs chmod 644
    chmod 644 justfile README.md packages/fluent-voice/examples/stt.rs packages/fluent-voice/examples/tts.rs
    @echo "All Cargo.toml files and protected files are now writable"

# Show status of Cargo.toml file permissions
status:
    @echo "Cargo.toml file permissions:"
    @rg --files --type toml | rg '^.*Cargo\.toml$' | xargs ls -l | awk '{print $1, $9}'
    @echo "Protected file permissions:"
    @ls -l justfile README.md packages/fluent-voice/examples/stt.rs packages/fluent-voice/examples/tts.rs | awk '{print $1, $9}'

# Format and check across entire workspace
check:
    cargo fmt --all
    cargo check --workspace --all-targets --all-features --message-format short --quiet
    cargo clippy --workspace --all-targets --all-features --message-format short --quiet

# Test across entire workspace  
test:
    cargo fmt --all
    cargo nextest run --workspace --message-format short --quiet

# Build release across entire workspace
build:
    cargo fmt --all
    cargo build --workspace --release --message-format short --quiet

# Regenerate workspace-hack after dependency changes
hakari:
    cargo hakari generate
    cargo hakari verify

# Complete workspace-hack regeneration from scratch
hakari-regenerate:
    #!/usr/bin/env bash
    set -euo pipefail
    
    echo "🔧 Commenting out workspace-hack dependencies in all packages..."
    # Comment out workspace-hack dependencies in all package Cargo.toml files
    find ./packages -name "Cargo.toml" -exec sed -i.bak 's/^fluent-voice-workspace-hack = /# fluent-voice-workspace-hack = /' {} \;
    
    echo "🔧 Commenting out workspace-hack member in root Cargo.toml..."
    # Comment out workspace-hack member in root Cargo.toml
    sed -i.bak 's/^    "workspace-hack",/    # "workspace-hack",/' ./Cargo.toml
    
    echo "🔧 Backing up hakari configuration..."
    # Backup current hakari config
    if [ -f .config/hakari.toml ]; then
        mv .config/hakari.toml .config/hakari.toml.bk
    fi
    
    echo "🔧 Removing existing workspace-hack..."
    # Remove existing workspace-hack
    rm -rf ./workspace-hack
    
    echo "🔧 Initializing fresh workspace-hack..."
    # Initialize new workspace-hack (provide automatic yes input)
    echo "y" | cargo hakari init ./workspace-hack
    
    echo "🔧 Renaming package to fluent-voice-workspace-hack..."
    # Update package name in workspace-hack/Cargo.toml (using perl for cross-platform compatibility)
    perl -i.bak -pe 's/name = "workspace-hack"/name = "fluent-voice-workspace-hack"/' ./workspace-hack/Cargo.toml
    
    echo "🔧 Uncommenting workspace-hack member in root Cargo.toml..."
    # Uncomment workspace-hack member in root Cargo.toml
    sed -i.bak 's/^    # "workspace-hack",/    "workspace-hack",/' ./Cargo.toml
    
    echo "🔧 Uncommenting workspace-hack dependencies..."
    # Uncomment workspace-hack dependencies in all package Cargo.toml files
    find ./packages -name "Cargo.toml" -exec sed -i.bak 's/^# fluent-voice-workspace-hack = /fluent-voice-workspace-hack = /' {} \;
    
    echo "🔧 Generating hakari configuration..."
    # Generate hakari
    cargo hakari generate
    
    echo "🔧 Verifying hakari configuration..."
    # Verify hakari
    cargo hakari verify
    
    echo "🔧 Cleaning up backup files..."
    # Clean up backup files
    find ./packages -name "Cargo.toml.bak" -delete
    rm -f ./workspace-hack/Cargo.toml.bak
    rm -f ./Cargo.toml.bak
    
    echo "✅ Workspace-hack regeneration complete!"