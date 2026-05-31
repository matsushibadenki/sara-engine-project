# Release Checklist

## Pre-Release Validation

1. Run the focused regression suite:
   `pytest -q tests/test_release_soak.py tests/test_sara_cli_dispatch.py tests/test_cli_entrypoints.py tests/test_inference_reliability.py tests/test_inference_memory_io.py tests/test_spiking_llm_memory_io.py tests/test_direct_map_utils.py tests/test_chat_agent_calculator.py tests/test_sara_agent_dialogue.py tests/test_practical_reliability.py`
   For Phase 3/4 and operational gate changes, also run:
   `pytest -q tests/test_phase3_accuracy_benchmarks.py tests/test_phase4_operational_cycle.py tests/test_operational_readiness.py tests/test_release_soak.py tests/test_release_gate.py`
2. Run the wall-clock soak script:
   `python scripts/eval/release_soak.py --duration-seconds 5 --min-agent-turns 24 --min-inference-iterations 32`
   To embed the latest Phase 3 accuracy summary into the managed soak report:
   `python scripts/eval/release_soak.py --duration-seconds 5 --min-agent-turns 24 --min-inference-iterations 32 --include-accuracy`
   For the final shipping decision, run the extended profile:
   `python scripts/eval/release_soak.py --profile extended`
3. Run the automated release gate:
   `python scripts/eval/release_gate.py`
   Confirm the Phase 3 accuracy summary is written under:
   `workspace/evaluation/phase3_accuracy_summary.txt`
   For practical deployment decisions, run the unified operational gate:
   `python scripts/eval/operational_readiness.py`
   For staging rehearsal with fresh artifacts:
   `python scripts/eval/operational_readiness.py --refresh-artifacts --soak-profile extended --include-accuracy`
   For final production promotion with strict extended requirements:
   `python scripts/eval/operational_readiness.py --strict-production`
   To generate a full operational report, runbook, and action manifest from fresh artifacts:
   `python scripts/eval/operational_readiness.py --refresh-artifacts --soak-profile extended --include-accuracy --strict-production`
   To verify the Phase 4 release/extended cycle plan:
   `python scripts/eval/phase4_operational_cycle.py --dry-run`
   To run the full Phase 4 operational cycle:
   `python scripts/eval/phase4_operational_cycle.py`
   For final v1.1 promotion gate:
   `python scripts/eval/v1_release_gate.py`
4. Confirm the soak report is written under:
   `workspace/release/release_soak_report.json`
   Confirm the human-readable summary is written under:
   `workspace/release/release_soak_summary.txt`
   Review the summary `overall_status` and section-level `PASS/WARN` markers before proceeding.
   If `Gate` shows `WARN`, review the listed errors before running the standalone release gate.
   If accuracy is embedded, also review the `Phase 3 Focus` section for `few_shot` and `continual` status.
   Also confirm `Stage B` shows `PASS`, including the world-model prototype minimums for transition, command, predictor snapshot, runtime tracking, and shift tracking readiness.
   Also confirm the Stage B operator/speculative fields are present in the summary:
   `stage_b_operator_coverage_ready == True`
   `stage_b_operator_consistency_ready == True`
   `stage_b_counterfactual_viability_ready == True`
   `stage_b_speculative_acceptance_ready == True`
   `stage_b_speculative_rollback_ready == True`
   In the `Predictive Detail` section, review:
   `transition_operator`
   `speculative_predicted_operator`
   `speculative_verified_operator`
   `speculative_operator_match`
   `speculative_acceptance`
   `speculative_rollback_observable`
   `speculative_counterfactual_viable`
   In the `Inference` section, confirm the runtime ratios remain healthy:
   `runtime_operator_consistency_ratio`
   `runtime_speculative_acceptance_ratio`
   `runtime_speculative_rollback_ratio`
   `runtime_counterfactual_viability_ratio`
   The report should now also include `release_metadata` so version alignment, console scripts, and release notes context can be reviewed from one place.
   Confirm operational readiness artifacts are present when using the unified gate:
   `workspace/release/operational_readiness_report.json`
   `workspace/release/operational_readiness_summary.txt`
   `workspace/release/operational_readiness_runbook.md`
   `workspace/release/operational_readiness_runbook_actions.json`
   `workspace/release/operational_repair_plan.json`
   For Phase 4 cycle runs, confirm:
   `workspace/release/phase4_operational_cycle_report.json`
   `workspace/release/phase4_operational_cycle_summary.txt`
5. Confirm the soak report records:
   `agent.turns >= 24`
   `agent.issue_count == 0`
   `agent.history_bounded == true`
   `inference.iterations >= 32`
   `inference.roundtrip_ok == true`
   `inference.tuple_keys_only == true`
6. Review current pre-release notes:
   `doc/RELEASE_NOTES.md`
7. Confirm the active documentation split is current:
   `doc/IMPLEMENTED_FEATURES.md` contains completed v1.1 features
   `doc/ROADMAP.md` contains only post-v1.1 direction
   historical material is under `doc/old/`

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
- embedded or referenced Phase 3 accuracy satisfies both `stage_a_acceptance` and `stage_b_readiness`, including the world-model prototype minimum checks
- Phase 3 completion passes with `completion_score >= 1.0` and no failed `checks` entries
- Phase 4 completion passes with required metrics, `threshold_results`, and `quality_metrics` for retention, latency, hippocampal transfer, structural stability, and continual recovery
- Sparse diffusion block readiness passes and is present in the Phase 5 completion gate artifact
- `workspace/evaluation/energy_measurement_session_plan.json` is present and the research product completion gate reports `energy_measurement_session_plan: PASS`
- Stage B operator/speculative trace fields are present and do not indicate silent divergence between predicted and verified transition operators
- final shipping decisions use an `extended` soak profile report, not only the default release profile
- strict operational readiness and the Phase 4 operational cycle have been reviewed
- `python scripts/eval/research_product_completion_gate.py` passes before the final v1.1 release gate
- release notes are updated for the current version
- implemented feature inventory and roadmap are updated for the current release state
- `python scripts/eval/v1_release_gate.py` reports `target_version: 1.1.0`
- packaging metadata is internally consistent
