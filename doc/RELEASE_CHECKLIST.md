# Release Checklist

## Pre-Release Validation

1. Run the focused regression suite:
   `pytest -q tests/test_release_soak.py tests/test_sara_cli_dispatch.py tests/test_inference_reliability.py tests/test_inference_memory_io.py tests/test_spiking_llm_memory_io.py tests/test_direct_map_utils.py tests/test_chat_agent_calculator.py tests/test_sara_agent_dialogue.py tests/test_practical_reliability.py`
2. Run the wall-clock soak script:
   `python scripts/eval/release_soak.py --duration-seconds 5`
3. Confirm the soak report is written under:
   `workspace/release/release_soak_report.json`
4. Review current pre-release notes:
   `doc/RELEASE_NOTES.md`

## Packaging Checks

1. Confirm `pyproject.toml` and `Cargo.toml` versions match.
2. Confirm Python entry points still expose:
   `sara-chat`
   `sara-train`
3. Confirm managed output paths are respected:
   no generated artifacts in repository root
   reports under `workspace/`
   model files under `models/`

## Operational Checks

1. Verify agent diagnostics remain bounded and do not grow unbounded during repeated turns.
2. Verify inference memory round-trip preserves tuple keys and integer token ids.
3. Verify legacy scripts are treated as non-production references only.

## Release Gate

Do not mark the build as full production release unless:

- regression tests pass
- soak report shows bounded state and successful memory round-trip
- release notes are updated for the current version
- packaging metadata is internally consistent
