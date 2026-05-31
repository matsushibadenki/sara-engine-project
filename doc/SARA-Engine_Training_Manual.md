# SARA Engine Data Pipeline And Training Manual

This manual covers the active data, training, inference, and memory-maintenance path. It should be read together with [policy.md](policy.md) and [TOOLS.md](TOOLS.md).

Run commands from the repository root.

## Data Management

SARA Engine stores training material in `data/sara_corpus.db` and exports managed training files under `data/raw/` and `data/processed/`.

Initialize or reset the corpus database:

```bash
python scripts/sara_cli.py db-reset
```

Import text or chat JSONL:

```bash
python scripts/sara_cli.py db-import data/raw/example.txt --category document --lang en
python scripts/sara_cli.py db-import data/raw/chat_data.jsonl --category dialogue --lang ja
```

Review imported material:

```bash
python scripts/sara_cli.py db-status
python scripts/sara_cli.py db-status --format json
python scripts/sara_cli.py db-list --limit 20
```

Control which materials are active:

```bash
python scripts/sara_cli.py db-activate --category document --min-quality-score 0.8
python scripts/sara_cli.py db-deactivate --source old_source
```

Export active data for training:

```bash
python scripts/sara_cli.py db-export
```

Expected managed outputs include:

- `data/processed/corpus.txt`
- `data/raw/chat_data.jsonl`
- optional reports under `workspace/reports/`

## Training Paths

### Self-Organized SNN Training

This is the preferred policy-aligned path. It uses SNN-native memory and local learning behavior rather than runtime backpropagation.

```bash
python scripts/sara_cli.py train-self-org
```

The model output is stored under `models/`.

### Staged Real-Data Curriculum (Small / Medium / Large)

For practical progression from pilot data to production-scale data, use the staged curriculum runner:

```bash
python scripts/sara_cli.py train-curriculum --stage small --dry-run
python scripts/sara_cli.py train-curriculum --stage small --preflight-only
python scripts/sara_cli.py train-curriculum --stage medium
python scripts/sara_cli.py train-curriculum --stage large
```

Behavior by stage:

- `small`: quality-focused export + self-org + SNN-LM + Phase 3/5 gates.
- `medium`: same training path + Phase 4 continual/scale completion gates.
- `large`: same as medium + strict operational readiness for shipping-grade validation.
- `preflight-only`: checks corpus DB presence, active material count, quality-filtered export count, and stage recommended scale before spending training time.

Managed report outputs:

- `workspace/reports/real_data_curriculum_small.json`
- `workspace/reports/real_data_curriculum_medium.json`
- `workspace/reports/real_data_curriculum_large.json`

### Legacy Distilled Agent Memory

This path remains available for compatibility and small conversational-memory experiments.

```bash
python scripts/sara_cli.py train-distill --model models/sara_agent
```

Treat this as a legacy helper path. Production work should prefer the SNN-native and release-gated paths.

### Subword SNN Language Model

For subword-level SNN language-model experiments:

```bash
python scripts/train/train_snn_lm.py --corpus data/processed/corpus.txt --save-dir models/snn_lm_pretrained
```

Optional chat data can be included:

```bash
python scripts/train/train_snn_lm.py --corpus data/processed/corpus.txt --chat-data data/raw/chat_data.jsonl --save-dir models/snn_lm_pretrained
```

## Inference And Chat

Chat with the self-organized model:

```bash
python scripts/sara_cli.py chat-self-org
```

Chat with the legacy distilled agent state:

```bash
python scripts/sara_cli.py chat-distill --model models/sara_agent
```

Chat with the subword SNN model:

```bash
python scripts/eval/chat_snn_lm.py --model-dir models/snn_lm_pretrained
python scripts/eval/chat_snn_lm.py --model-dir models/snn_lm_pretrained --debug
```

## Memory Maintenance

Inspect a saved memory artifact:

```bash
python scripts/sara_cli.py inspect-memory
```

Upgrade an older memory artifact into the current managed format:

```bash
python scripts/sara_cli.py upgrade-memory
```

Build replay data for upgrade or recovery workflows:

```bash
python scripts/sara_cli.py build-replay-data --data data/raw/chat_data.jsonl
```

Prune low-value memory weights:

```bash
python scripts/sara_cli.py prune --threshold 50.0
```

Clean interim and processed data outputs:

```bash
python scripts/sara_cli.py clean
```

## Quality Gates After Training

After training or memory changes, run at least the relevant lightweight gates:

```bash
python scripts/eval/phase3_accuracy_suite.py
python scripts/eval/phase3_completion_gate.py
python scripts/eval/phase4_scale_continual_benchmark.py
python scripts/eval/phase4_completion_gate.py
```

For release-oriented validation:

```bash
python scripts/eval/operational_readiness.py --refresh-artifacts --soak-profile extended --include-accuracy --strict-production
python scripts/eval/phase4_operational_cycle.py --dry-run
```

Use [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) for final release review.
