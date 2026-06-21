# SARA/ANN Physical Energy Measurement Protocol

## Objective

Measure SARA and an ANN baseline on the same bounded task and report:

`joule_per_success = measured_joules / successful_trials`

An energy advantage is valid only when the paired systems satisfy the same machine-checkable success criterion and the configured success-rate parity threshold.

## Fixed Conditions

For every SARA/ANN pair, keep these values identical:

- Physical CPU and host
- CPU thread count and process affinity
- Power mode and AC/battery state
- Task fixture hash and corpus/index contents
- Success criterion ID
- Measurement boundary
- Measurement tool and sampling method
- Warm-up count
- Measured repetitions
- Trial count

Record these fields through `record-energy-measurement`. The readiness evaluator rejects a pair when any fixed field differs.

## Run Design

1. Stop unrelated CPU-intensive work and record the measurement tool version.
2. Fix CPU threads, affinity, and power mode.
3. Hash the frozen task fixture and assign a success criterion ID.
4. Choose a measurement boundary, such as `warm-index-query-only-v1`.
5. Run the declared warm-up count without recording it as measured work.
6. Measure at least three paired replicates per task.
7. Alternate order by replicate:
   - replicate 1: SARA then ANN
   - replicate 2: ANN then SARA
   - replicate 3: SARA then ANN
8. Use the same `pair_id` and `replicate_index` for both systems in one pair.
9. Record invalid or interrupted runs separately; do not silently select the best run.

## Required Task Types

- Retrieval with frozen corpus, candidates, queries, and success rules.
- Continual adaptation with frozen event stream, adaptation opportunity, and post-adaptation success rules.

## Aggregation

- Compute `joule_per_success` for every system row.
- Validate the SARA/ANN pair before using it.
- Compute the ANN/SARA ratio for every valid replicate.
- Report the per-task median and median absolute deviation.
- Report task-level losses; do not hide them in an aggregate.
- Do not claim an energy win when quality parity fails.

## Managed Artifacts

- Raw rows: `data/raw/energy_measurements.jsonl`
- Readiness report: `workspace/evaluation/energy_measurement_readiness.json`
- Readiness summary: `workspace/evaluation/energy_measurement_readiness_summary.txt`
- Session plan: `workspace/evaluation/energy_measurement_session_plan.json`
- Session plan summary: `workspace/evaluation/energy_measurement_session_plan.txt`

## Commands

Generate or refresh the session plan:

```bash
python scripts/eval/energy_measurement_readiness.py
```

Prepare a dry-run pair manifest:

```bash
python scripts/sara_cli.py run-physical-energy-pair \
  --pair-id retrieval-r1 \
  --replicate-index 1 \
  --dry-run
```

Execute the frozen pair without recording energy rows:

```bash
python scripts/sara_cli.py run-physical-energy-pair \
  --pair-id retrieval-r1 \
  --replicate-index 1 \
  --measurement-tool external-meter-manual-v1
```

The runner fixes the corpus/task hash, success criterion, CPU identity, thread environment, repetitions, warm-up count, and alternating run order. It writes workload results and timing traces under `workspace/evaluation/`. Supply measured SARA and ANN joules only when both measurements correspond to that exact manifest.

Record each measured row with the command template written to the generated session plan. Replace all placeholders with observed or frozen protocol values.
