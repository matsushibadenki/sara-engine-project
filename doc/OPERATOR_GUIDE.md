# SARA Operator Guide

This is the short operational path for researchers and release operators. Commands are CPU-first and write reports only under `workspace/`.

## Daily Review

```bash
python scripts/sara_cli.py eval-operator-dashboard
```

Review `workspace/evaluation/operator_dashboard_summary.txt` first. It lists artifact status, the next command, proven evidence, and explicitly unproven claims.

## Reproduce Evidence

```bash
python scripts/sara_cli.py eval-research-benchmark-suite
python scripts/sara_cli.py eval-phase9-completion
python scripts/sara_cli.py eval-phase10-completion
python scripts/sara_cli.py eval-phase11-completion
python scripts/sara_cli.py eval-phase13-completion
```

Physical energy measurement is indefinitely pending and is not part of the active execution queue. Do not run a meter session or generate joule rows unless the operator explicitly reopens Phase 6.

If Phase 6 is explicitly reopened, prepare the paired session without fabricating joules:

```bash
python scripts/sara_cli.py run-physical-energy-session-batch
python scripts/sara_cli.py eval-physical-energy-session-progress
```

## Troubleshooting

- **Python version:** use the project Python 3.10 environment. Check with `python --version`; optional Rust builds use `python -m maturin develop --features extension-module`.
- **Missing optional dependency:** run the command that reports the missing dependency. Optional ANN, LLM, hardware, and meter integrations must not be replaced with fabricated evidence.
- **Managed output violation:** write temporary files under `workspace/`; training data belongs under `data/`; final model artifacts belong under `models/`. Do not redirect reports to the repository root.
- **Gate failure:** open the referenced JSON report, run its suggested command, and rerun `eval-operator-dashboard`.
- **Physical energy indefinitely pending:** keep Phase 6 labeled pending and non-blocking. CPU telemetry, `ioreg`, wall-clock time, event counts, and proxy costs must never be promoted to physical joules.
