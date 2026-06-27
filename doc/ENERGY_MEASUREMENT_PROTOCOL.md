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
- Physical session batch plan: `workspace/evaluation/physical_energy_session_batch.json`
- Physical session batch summary: `workspace/evaluation/physical_energy_session_batch.txt`
- Physical session progress report: `workspace/evaluation/physical_energy_session_progress.json`
- Physical session progress summary: `workspace/evaluation/physical_energy_session_progress.txt`

## Commands

Generate or refresh the session plan:

```bash
python scripts/eval/energy_measurement_readiness.py
```

Expand the session plan into concrete pair runs:

```bash
python scripts/sara_cli.py run-physical-energy-session-batch
```

Optionally dry-run every frozen pair command in the batch:

```bash
python scripts/sara_cli.py run-physical-energy-session-batch \
  --execute-dry-run-pairs
```

Summarize session progress from the batch plan and recorded rows:

```bash
python scripts/sara_cli.py eval-physical-energy-session-progress
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

Record a completed external-meter session from a small managed JSON file:

```json
{
  "schema": "sara-physical-meter-readings-v1",
  "pair_id": "retrieval-r1",
  "replicate_index": 1,
  "readings": {
    "sara": {
      "joules": 12.34
    },
    "ann": {
      "average_watts": 4.2,
      "duration_seconds": 3.5
    }
  }
}
```

Then append both rows from the frozen pair:

```bash
python scripts/sara_cli.py run-physical-energy-pair \
  --pair-id retrieval-r1 \
  --replicate-index 1 \
  --measurement-tool external-meter-manual-v1 \
  --meter-reading-path workspace/evaluation/retrieval-r1_meter.json
```

The meter JSON may use either direct `joules` or `average_watts * duration_seconds` per system. The runner rejects pair or replicate mismatches before writing `data/raw/energy_measurements.jsonl`.

For SARA runs that keep persistent self-state and idle replay enabled, the frozen workload also exports bounded maintenance observations:

- `maintenance_selected_count`
- `maintenance_phase_count`
- `maintenance_refresh_count`
- `maintenance_event_cost`
- `maintenance_idle_self_state_ok_count`
- `maintenance_spontaneous_event_count`
- `maintenance_predicted_event_count`

Treat these as internal-activity observability fields, not as joules. They are recorded beside the physical energy row so later analysis can distinguish "lower joules because less useful work happened" from "lower joules despite sustained internal self-maintenance".

Additional pair artifacts:

- Pair manifest: `workspace/evaluation/physical_energy_pair_manifest.json`
- Pair trace: `workspace/evaluation/physical_energy_pair_trace.jsonl`
- Pair report: `workspace/evaluation/physical_energy_pair_report.json`
- Pair summary: `workspace/evaluation/physical_energy_pair_summary.txt`
- Pair meter template: `workspace/evaluation/physical_energy_pair_meter_template.json`

The pair report now includes:

- `resume_append_command_template`: rerun the same frozen pair with `--sara-joules` and `--ann-joules` filled in.
- `record_measurement_commands`: per-system `record-energy-measurement` commands with frozen fairness fields, trial counts, durations, run order, and maintenance fields already filled in. Only the exact joule value remains to be inserted.

The generated meter template is the preferred starting point for manual or external-meter capture. It already contains the frozen `pair_id`, `replicate_index`, measurement metadata, run order, and observed workload durations when they are available.

The readiness session plan now carries both command families:

- `command_template`: append-ready `record-energy-measurement`
- `pair_command_template`: `run-physical-energy-pair` with a planned `pair_id`, replicate placeholder, and managed `meter_template_path`

The physical session batch report collapses the duplicated per-system session-plan rows into concrete pair-level runs. Each batch item expands the replicate placeholder and emits one frozen `run-physical-energy-pair` command plus its managed manifest, trace, report, summary, and meter-template paths.

The physical session progress report then compares those planned pair runs against `data/raw/energy_measurements.jsonl` and labels each expected pair as `complete_valid_pair`, `partial_pair`, `invalid_pair`, or `missing_pair`. This is the quickest way to see whether a laboratory session is actually complete before regenerating the full readiness report.

`python scripts/eval/energy_measurement_readiness.py` now also emits the same session-progress artifact family directly, derived from its managed session plan and current measurement rows. This keeps the readiness report, session plan, and session progress synchronized even when the separate batch helper is not used.

Record each measured row with the command template written to the generated session plan. Replace all placeholders with observed or frozen protocol values.
