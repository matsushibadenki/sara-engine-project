# Project Output Policy

This repository uses fixed output locations. New code must not create files outside these managed directories.

## Allowed locations

- Training data must live only under `data/`.
- `data/raw/` is for collected source data and exported chat JSONL files.
- `data/processed/` is for cleaned corpora and finalized processed datasets.
- `data/interim/` is for temporary data artifacts created during preprocessing.
- Working files, temporary scratch files, and non-final artifacts must live only under `workspace/`.
- Final model artifacts must live only under `models/`.

## Disallowed locations

- Do not write generated files to the repository root.
- Do not create ad hoc output directories outside `data/raw`, `data/processed`, `data/interim`, `workspace`, or `models`.
- Do not save completed model artifacts under `workspace/`.

## Implementation rule

- Prefer using `src/sara_engine/utils/project_paths.py` for all new read/write paths.
- When adding a new output path, validate it against the managed directories before writing.
