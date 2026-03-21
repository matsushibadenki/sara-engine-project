# Release Notes

## Current Pre-Release

This pre-release focuses on production hardening for the CPU-first SNN stack.

### Highlights

- Unified direct-memory serialization and restoration across `SpikingLLM`, `SaraInference`, training scripts, evaluation scripts, and maintenance utilities.
- Removed unsafe `eval()` usage from the active loading path and replaced it with safe parsers and shared helpers.
- Added lightweight runtime diagnostics for `SaraAgent`, including session persistence of recent issues and optional CLI diagnostics display.
- Replaced the chat calculator `eval()` path with a safe arithmetic parser that supports only basic operations.
- Added stronger FORCE artifact validation to fail fast on malformed shapes and mismatched metadata.
- Fixed the `UnifiedSNNModel` FORCE readout path so one logical step updates reservoir state only once.

### Reliability Work Included

- Added regression tests for direct-memory round-trips, inference memory compatibility, agent session recovery, CLI dispatch, and calculator safety.
- Added lightweight soak tests for repeated `SaraAgent` dialogue turns and repeated `SaraInference` memory updates.
- Marked `scripts/old/` as legacy and documented that it is not the recommended production path.

### Operational Notes

- The project continues to prioritize SNN-friendly efficiency: no backpropagation, no required GPU path, and no matrix-heavy runtime dependency for the newly added reliability features.
- New diagnostics are intentionally lightweight and bounded to avoid turning observability into a hidden energy cost.

### Known Gaps Before Full Production Release

- Longer wall-clock soak runs outside unit-test scale are still recommended.
- End-to-end CLI scenario coverage is improved but not yet exhaustive across all commands.
- Legacy scripts remain available for reference, but production usage should prefer `src/sara_engine`, `scripts/train`, `scripts/eval`, and `scripts/sara_cli.py`.
