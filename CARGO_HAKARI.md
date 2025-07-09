# The Complete Guide to cargo-hakari for Rust Dependency Management

## What is cargo-hakari and why it matters

Cargo-hakari is a command-line tool that automates the creation and maintenance of "workspace-hack" crates to dramatically improve build times in Rust workspaces. By unifying dependency features across all workspace members, it eliminates redundant compilation of the same dependencies with different feature sets, achieving up to **100x speedups for individual commands** and **1.7x cumulative build time improvements**.

### The problem it solves

In Rust workspaces, different crates often depend on the same external library but with different feature sets:

```toml
# crate-a/Cargo.toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }

# crate-b/Cargo.toml  
[dependencies]
serde = { version = "1.0", features = ["rc"] }

# crate-c/Cargo.toml
[dependencies]
serde = { version = "1.0", features = ["derive", "rc", "alloc"] }
```

Without hakari, Cargo may compile serde multiple times with different feature combinations, leading to slower builds and increased disk usage. Cargo-hakari solves this by creating a unified workspace-hack crate that forces consistent feature resolution across all builds.

## Installing cargo-hakari

### Primary installation method
```bash
cargo install cargo-hakari --locked
```

### Alternative installation methods
```bash
# Using cargo-binstall (faster)
cargo binstall cargo-hakari

# In GitHub Actions
- name: Install cargo-hakari
  uses: taiki-e/install-action@v2
  with:
    tool: cargo-hakari

# Download pre-built binaries
# Visit: https://github.com/guppy-rs/guppy/releases
```

**System Requirements:**
- Rust 1.82 or later (MSRV)
- Full support on Unix/Linux/macOS
- Windows supported (with forward slash paths)

## Configuring and initializing hakari

### Step 1: Prerequisites
```bash
# Add Cargo.lock to version control (required)
git add Cargo.lock
git commit -m "Add Cargo.lock for hakari"

# Ensure resolver v2 in workspace Cargo.toml
[workspace]
resolver = "2"
```

### Step 2: Initialize workspace-hack
```bash
# Initialize with default name
cargo hakari init

# Or specify custom path
cargo hakari init --path my-workspace-hack

# Non-interactive mode (for CI/automation)
cargo hakari init workspace-hack --yes
```

### Step 3: Configure hakari
Create or edit `.config/hakari.toml`:

```toml
# Basic configuration
hakari-package = "workspace-hack"
resolver = "2"
dep-format-version = "4"

# Platform optimization
platforms = [
    "x86_64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-msvc",
]

# Advanced options
unify-target-host = "auto"  # Best performance
output-single-feature = false

# Exclusions
[traversal-excludes]
workspace-members = ["test-crate"]
third-party = [
    { name = "criterion" },  # Benchmark-only
]

[final-excludes]
third-party = [
    { name = "fail" },  # Common test crate
]

# Custom registries
[registries]
internal = { index = "https://internal.example.com/index" }
```

### Step 4: Generate and apply
```bash
# Generate workspace-hack content
cargo hakari generate

# Add workspace-hack dependency to all crates
cargo hakari manage-deps
```

## Using hakari for dependency management

### Adding new dependencies

1. Add to your crate's `Cargo.toml`:
```toml
[dependencies]
reqwest = { version = "0.11", features = ["json"] }
```

2. Update hakari:
```bash
cargo hakari generate
cargo hakari manage-deps
```

### Updating dependencies

```bash
# Update all dependencies
cargo update

# Update specific dependency
cargo update -p serde

# Update to specific version
cargo update -p tokio --precise 1.35.0

# Regenerate hakari after updates
cargo hakari generate
```

### Upgrading to new major versions

1. Edit `Cargo.toml`:
```toml
[dependencies]
clap = "4.0"  # was "3.0"
```

2. Update and regenerate:
```bash
cargo update -p clap
cargo hakari generate
```

### Removing dependencies

1. Remove from `Cargo.toml`
2. Regenerate hakari:
```bash
cargo hakari generate
cargo hakari manage-deps
```

## Best practices and common patterns

### Workspace organization patterns

**For library-focused workspaces:**
```toml
# .config/hakari.toml
output-single-feature = false  # Default, conservative
[final-excludes]
workspace-members = ["examples", "benches"]
```

**For application-focused workspaces:**
```toml
# .config/hakari.toml
output-single-feature = true  # More aggressive unification
platforms = ["x86_64-unknown-linux-gnu"]  # Production target only
```

### When to regenerate

Always regenerate after:
- Adding/removing dependencies
- Changing feature flags
- Updating dependency versions
- Modifying hakari configuration
- Merging branches with dependency changes

### CI/CD integration

```yaml
# .github/workflows/hakari.yml
name: Verify workspace-hack
on: [push, pull_request]

jobs:
  hakari-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Install cargo-hakari
        uses: taiki-e/install-action@v2
        with:
          tool: cargo-hakari
      - name: Check workspace-hack is up-to-date
        run: cargo hakari generate --diff
      - name: Verify all crates depend on workspace-hack
        run: cargo hakari manage-deps --dry-run
```

### Handling merge conflicts

Never manually resolve conflicts in workspace-hack/Cargo.toml:

```bash
# After merge conflict
git checkout --theirs Cargo.lock  # Or resolve Cargo.lock first
cargo hakari generate  # Regenerate from scratch
git add workspace-hack/Cargo.toml
git commit
```

## How hakari works under the hood

### The three-phase algorithm

**Phase 1: Build Simulation**
- Simulates builds for every workspace package
- Tests with no features, default features, and all features
- Captures results for each configured platform
- Uses guppy's Cargo build simulator

**Phase 2: Feature Conflict Analysis**
- Identifies dependencies built with multiple feature sets
- Computes the union of all feature sets for conflicting dependencies
- Creates initial set of dependencies requiring unification

**Phase 3: Fixpoint Iteration**
- Iterates until reaching stable solution
- Checks if adding dependencies creates new conflicts
- Continues until no new conflicts arise

### Technical architecture

```rust
// Simplified algorithm representation
for package in workspace_packages {
    for platform in target_platforms {
        for features in [none, default, all] {
            simulate_build(package, platform, features);
            record_dependency_features();
        }
    }
}

// Unify conflicting features
for (dep, feature_sets) in dependencies {
    if feature_sets.len() > 1 {
        unified_features = union_all(feature_sets);
        workspace_hack.add(dep, unified_features);
    }
}
```

## Common issues and troubleshooting

### Build errors after setup
**Solution:** Ensure `resolver = "2"` in workspace Cargo.toml

### Dependencies out of sync
```bash
# Verify state
cargo hakari verify

# Fix by regenerating
cargo hakari generate
cargo hakari manage-deps
```

### Publishing issues
```bash
# Option 1: Use hakari publish
cargo hakari publish -p my-crate

# Option 2: Temporarily disable
cargo hakari disable
cargo publish -p my-crate
cargo hakari generate
```

### Debugging commands
```bash
# Explain why dependency is included
cargo hakari explain serde

# Show what would change
cargo hakari generate --diff

# Verify configuration
cargo hakari verify
```

## CI/CD pipeline integration

### GitHub Actions complete example
```yaml
name: Rust CI with hakari
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        rust: [stable, beta]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}
      - uses: taiki-e/install-action@v2
        with:
          tool: cargo-hakari
      
      # Verify hakari state
      - name: Check workspace-hack
        run: |
          cargo hakari generate --diff
          cargo hakari manage-deps --dry-run
      
      # Regular CI tasks
      - name: Build
        run: cargo build --all-targets
      - name: Test
        run: cargo test --all
```

### GitLab CI example
```yaml
stages:
  - verify
  - test

hakari-verify:
  stage: verify
  image: rust:latest
  script:
    - cargo install cargo-hakari
    - cargo hakari generate --diff
    - cargo hakari manage-deps --dry-run
```

## Comparison with other approaches

### Feature comparison matrix

| Approach | Build Speed | Setup Complexity | Publishing Support | Best For |
|----------|-------------|------------------|-------------------|----------|
| **cargo-hakari** | 1.1x-100x faster | Medium | Complex | Large workspaces |
| **Standard workspaces** | Baseline | Low | Simple | Small projects |
| **cargo-workspaces** | Baseline | Medium | Excellent | Multi-crate publishing |
| **Workspace inheritance** | Baseline | Low | Simple | Dependency consistency |

### When to use cargo-hakari

Use hakari when you have:
- Large workspaces (50+ crates)
- Many external dependencies
- Frequent CI builds
- Build times over 5 minutes
- Development team of 5+ people

### Migration strategy

1. Start with workspace dependency inheritance (Cargo 1.64+)
2. Add cargo-hakari when builds exceed 10 minutes
3. Use cargo-workspaces for complex publishing needs

## Latest features and updates (2025)

### Current version: 0.9.36
- Released: February 21, 2025
- Rust 1.85 support
- Sparse registry support (`sparse+https://...`)
- Performance improvements (33% faster computation)

### Recent improvements
- **Configuration**: Default path changed to `.config/hakari.toml`
- **Format version 4**: Alphabetically sorted dependencies
- **New commands**: `explain`, `verify`, `disable`
- **Better CI support**: Enhanced `--diff` and `--dry-run` options

### Performance benchmarks (2024-2025)
- **Cumulative speedup**: ~1.7x faster builds
- **Individual commands**: 1.1x to 100x faster
- **Disk usage**: ~1.4x reduction (55GB vs 78GB)
- **Best gains**: `cargo check` commands

### Future outlook
- Rust RFC 3692 proposes native Cargo feature unification
- Continued active development and maintenance
- Growing adoption in large Rust projects

## Complete command reference

```bash
# Initialize workspace-hack
cargo hakari init [--path <PATH>]

# Generate/update workspace-hack
cargo hakari generate [--diff] [--quiet]

# Manage dependencies
cargo hakari manage-deps [--dry-run] [--remove]

# Verify state
cargo hakari verify [--quiet]

# Explain inclusions
cargo hakari explain <CRATE> [--all]

# Disable temporarily
cargo hakari disable

# Publish with hakari
cargo hakari publish -p <CRATE> [--dry-run]
```

## Key takeaways for LLMs

1. **Primary purpose**: Optimize Rust workspace builds by unifying dependency features
2. **Performance impact**: Up to 100x faster individual builds, 1.7x cumulative improvement
3. **Setup process**: Initialize → Generate → Manage dependencies → Verify in CI
4. **Daily workflow**: Regenerate after any dependency changes
5. **Best practices**: Use resolver v2, configure platforms, integrate with CI
6. **Troubleshooting**: Most issues solved by regenerating workspace-hack
7. **When to use**: Large workspaces with many dependencies and frequent builds

This guide provides comprehensive, actionable information for effectively using cargo-hakari in Rust projects, with practical examples and clear instructions for every aspect of the tool.
