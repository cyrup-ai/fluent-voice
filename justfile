# ────────────────────────────── Workspace Commands ──────────────────────────────
default:
    @just --list

# Make all Cargo.toml files read-only across the workspace
lock:
    rg --files --type toml | rg '^.*Cargo\.toml$' | xargs chmod 444
    @echo "All Cargo.toml files are now read-only"

# Make all Cargo.toml files writable again
unlock:
    rg --files --type toml | rg '^.*Cargo\.toml$' | xargs chmod 644
    @echo "All Cargo.toml files are now writable"

# Show status of Cargo.toml file permissions
status:
    @echo "Cargo.toml file permissions:"
    @rg --files --type toml | rg '^.*Cargo\.toml$' | xargs ls -l | awk '{print $1, $9}'

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