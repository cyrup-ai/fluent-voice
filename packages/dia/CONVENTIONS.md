# CONVENTIONS

## Project Preparation for Coding

### Start Clean

- Always run `cargo fmt && cargo check --message-format short --quiet`
  - Fix any errors or warnings before proceeding
  - That way, you know you're starting on solid ground

### Get the Docs

- analyze Cargo.toml and the source code to see which libraries or core to successful coding.
- check ./docs/**/* to see if the docs are already there.
- If not, or if you need more information on advanced topics or idioms, use these  MCP tools to get the docs:
  - `context7`,
  - `github-mcp-server`
  - `firecrawl-mcp`
- Save any new information in docs/ for later use.
- Correct any inaccuracies in docs/ if you identify them.

## async Rust

- ❌ In repositories we own:
  - ❌ NEVER use `async_trait` or `async fn` in any crate, or public surface
  - ❌ NEVER return `Box<dyn Future>` or `Pin<Box<dyn Future>>` from client interfaces
- ✅ Provide synchronous interfaces that return an awaitable Stream or a wrapped Future that allow the caller to await the result when they are ready.
- ✅ ALWAYS: make this super easy by using our `cyrup-ai/async_task` crate and the deliciously sweet Rust sugars it provides.
- DO: check out and understand this crate before you do any other dev.

## `cargo` rules & `Cargo.toml`

- Always use the latest version of each dependency unless a noted exception has been approved. You can find the latest with:
- `cargo search {{package_id}} limit 1`
- use `cargo` to add, remove, update or upgrade packages.
- Learn `cargo edit` and `cargo workspace` and you'll be good to go.
- Lint (after EVERY change): `cargo fmt && cargo check --message-format short --quiet`
- Build: `cargo build`, Run: `cargo run`
- Test: Always use `nextest` and `cargo test`

## Error Handling

- Use Result<T,E> with custom errors
- No unwrap() except in tests or in cases where the error is handled explicitly
- Handle all Result/Option values explicitly

## Style & Structure

- Tests go in `tests/` directory and not co-occuring with src files
- Use `cyrup-ai/async_task` crate for async operations universally
- Do not write blocking code at all, ever, ever even for tests
  - `nextest` provides a great way to test async code
- No single file should be more than 300 lines long
  - Decompose once we exceed that max line limit
  - Logically decompose into modules that handle singular concerns
- Rust official style: snake_case for variables/functions
- Use `tracing` for logs with appropriate levels
- ❌ NO suppression of compiler or clippy warnings
- ✅ All code MUST pass `cargo check --message-format short --quiet` without exception.
  - fix all warnings, it's worth it to have a clean code base and they are most often truly problematic
  - fix all errors ... it is not anyone else's fault, we're a team and we all have to fix up after one another sometimes
  - Take the time to fix warnings and errors properly
  - DO NOT EVER:
    - comment out the code
    - disable modules
    - stub methods
    - leave comments like "In the future we'll code it right ..."
    - suppress warnings by renaming variables with underscores or by annotating them with `#[allow(dead_code)]` or other suppressing annotations

## Be a Software Artisan

-
- Focus on interface first.
  - Who is using this product? How can we make this drop in easy for them to adopt?
  - How are they using it? What is intuitive in this context?
  - Ask questions before making up features we don't need.
- WRITE THE *MINIMAL AMOUNT OF CODE* NEEDED TO IMPACT A CHANGE (but do it fully and correctly)
  - Do not add features that are not requested.
  - NEVER EVER ADD `examples/*` unless Dave asks for them.
  - DO ADD tests in nextest in `./tests`. Focus on the key elements that prove it is really working for the user of the software.
  - DO NOT say "it's all good" or "completed" unless you have **tested like an end-user** (ie. `cargo run` for a bin) and verified the feature.
  - DO NOT add more than one binary per crate.

## "REAL WORLD" Rules

- ✅ All warnings must be fully resolved, not suppressed.
- DO NOT use annotations to suppress warnings of any type.
- DO NOT use private _variable naming to hide warnings.
  - Unused code is either:
    1. a bug that needs implementation to function
    2. a half-assed non-production implementation
    3. a mess that makes it hard to read and understand
- NEVER leave **"TODO: in a real world situation ..."** or *"In production we'll handle this differently ..."* or similar suggestions.
- *WRITE PRODUCTION QUALITY CODE ALL THE TIME!*. The future is now. This IS production!! You are the best engineers I know. Rise up.
- ASK, ASK, ASK -- I love your initiative but writing full modules that are all wrong is costly and time consuming to cleanup. Just ask and don't assume anything. I'll hurry along when it's time :)

## SurrealDB (awesome)

- Use SurrealDB for all database operations
- The syntax in version 2.2.1 has changed significantly
- use `kv-surrealkv` local file databases and `kv-tikv` for clustered/distributed databases.
- use the appropriate table type for the job (document, relational, graph, time series, vector)
- use `surrealdb-migrations` version 2.2+ for perfectly versioned migrations. This is really essential for distributed file-based data.
- use our `cyrup-ai/surrealdb-client` to get up and running fast with elegant traits and base implementations.

## Preferred Software

- `dioxus` (pure Rust front-end)
- `axum` (elite tokio based server)
- `floneum/floneum` ask "Kalosm" (local agents with superpowers)
- `surrealdb` (swiss army knife of fast, indexed storage and ML support)
- `livekit` for open-source real-time audio/video
- `clap`, `ratatui`, `crossterm` ... just amazing cli tools
- `serde` for serialization/deserialization
- `tokio` for asynchronous programming and io bound parallelism
- `rayon` for cpu bound parallelism
- `nextest` for hella-fast and lovely test execution
- `chromiumoxide` for web browser automation
  D
