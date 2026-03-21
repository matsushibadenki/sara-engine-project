# Release Checklist

## Pre-Release Validation

1. Run the focused regression suite:
   `pytest -q tests/test_release_soak.py tests/test_sara_cli_dispatch.py tests/test_inference_reliability.py tests/test_inference_memory_io.py tests/test_spiking_llm_memory_io.py tests/test_direct_map_utils.py tests/test_chat_agent_calculator.py tests/test_sara_agent_dialogue.py tests/test_practical_reliability.py`
2. Run the wall-clock soak script:
   `python scripts/eval/release_soak.py --duration-seconds 5 --min-agent-turns 24 --min-inference-iterations 32`
   For the final shipping decision, run the extended profile:
   `python scripts/eval/release_soak.py --profile extended`
3. Run the automated release gate:
   `python scripts/eval/release_gate.py`
4. Confirm the soak report is written under:
   `workspace/release/release_soak_report.json`
5. Confirm the soak report records:
   `agent.turns >= 24`
   `agent.issue_count == 0`
   `agent.history_bounded == true`
   `inference.iterations >= 32`
   `inference.roundtrip_ok == true`
   `inference.tuple_keys_only == true`
6. Review current pre-release notes:
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
4. Before a full production tag, archive the `extended` soak report under `workspace/release/` and confirm it passed without runtime issues.

## Release Gate

Do not mark the build as full production release unless:

- regression tests pass
- soak report satisfies the minimum workload thresholds and shows bounded state
- final shipping decisions use an `extended` soak profile report, not only the default release profile
- release notes are updated for the current version
- packaging metadata is internally consistent
