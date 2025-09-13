# Improve Search Patterns for Unwrap Detection

## Description
Fix overly broad search patterns that flag legitimate APIs like `Arc::try_unwrap()` as violations when searching for problematic `.unwrap()` usage.

## Current Issue
Search term "unwrap" incorrectly flagged proper error handling in `packages/vad/src/vad.rs:158-159`:
```rust
Arc::try_unwrap(session).map_err(|_| Error::PredictionFailed("Cannot unwrap shared session".to_string()))?
```

## Technical Resolution
Use precise regex patterns to avoid false positives:

```bash
# ❌ OLD: Overly broad search  
grep -r "unwrap" --include="*.rs" ./packages/

# ✅ NEW: Precise regex patterns
grep -rE "\.unwrap\(\)" --include="*.rs" ./packages/  # Only matches .unwrap() calls
grep -rE "\.expect\(" --include="*.rs" ./packages/   # Only matches .expect() calls

# ✅ Exclude legitimate APIs
grep -rE "\.unwrap\(\)" --include="*.rs" ./packages/ | grep -v "try_unwrap\|unwrap_or"
```

## Success Criteria
- [ ] Update search patterns to use precise regex
- [ ] Exclude legitimate unwrap APIs (try_unwrap, unwrap_or, etc.)
- [ ] Verify no false positives for proper error handling
- [ ] Document improved search methodology
- [ ] Test patterns against known legitimate usage

## Dependencies
None - independent tooling improvement

## Architecture Impact
LOW - Improves code quality analysis accuracy