# SARA Engine Storage Format Strategy

This document defines the long-term storage direction for SARA Engine artifacts.

The project should not converge on a single universal file format. SARA has three different needs:

1. auditable append-only research records
2. deterministic manifests and validation metadata
3. compact runtime state for fast loading and bounded deployment

The long-term direction is therefore a **three-layer storage model**, not "everything as JSONL" and not "everything as binary".

## Design Goal

Use the most appropriate format for each artifact class while preserving:

- source lineage
- deterministic validation
- managed output-path policy
- CPU-first usability
- bounded runtime load cost
- future compatibility with sparse edge and neuromorphic export

## Current Assessment

The current repository is partly well-shaped already:

- `JSONL` is used heavily for training materials, candidate events, manifests, and traces.
- `JSON` is used for reports, summaries, manifests, and edge-export metadata.
- `msgpack` is already used for some inference-memory model artifacts.
- edge export already separates payload and integrity concerns with `format_version`, `format_capabilities`, and `edge_manifest`.

This is good enough for research iteration, but it is not yet a fully normalized long-term format strategy.

The main weakness is that numeric runtime state, audit-heavy research records, and operator summaries still use overlapping persistence styles without one canonical division of responsibility.

## Target Three-Layer Model

### Layer 1: Append-Only Event Records

Use append-only `JSONL` for records whose primary value is auditability, reviewability, lineage, or streaming append behavior.

Recommended examples:

- observed events
- candidate events
- verified relations
- curriculum rows
- repair logs
- collection targets
- learning-material rows
- source-backed latent rows
- replay selection traces
- operational action traces

Why:

- easy incremental writes
- resilient partial recovery
- line-by-line inspection
- git-friendly debugging
- schema-per-row evolution is easier than monolithic blob upgrades

Do not use this layer for large numeric tensors or high-frequency runtime state snapshots.

### Layer 2: Canonical Manifests

Use deterministic `JSON` manifests for package-level metadata and validation surfaces.

Recommended examples:

- dataset manifests
- benchmark manifests
- model package manifests
- edge manifests
- release action manifests
- schema/version declarations
- digest and capability metadata
- source-split and evaluation-isolation metadata

Every canonical manifest should prefer:

- explicit `schema`
- explicit `version`
- artifact counts
- digest or checksum fields
- source fingerprints
- dependency references
- migration hints when needed

Why:

- deterministic validation
- easy operator inspection
- stable integration with CLI tools and tests
- natural place for integrity, provenance, and compatibility metadata

### Layer 3: Compact Runtime State Payloads

Use compact binary state for artifacts whose primary value is load efficiency, bounded memory use, or deployment portability.

Recommended examples:

- sparse readout weights
- routing tables
- associative memory state
- cache snapshots
- replay indices
- persistent self-state snapshots
- compact edge deployment payloads

Preferred current direction:

- continue using `msgpack` or a similarly compact binary container for runtime-oriented state
- keep the binary payload wrapped by a nearby deterministic manifest

Why:

- smaller file size
- faster load and save
- less numeric serialization overhead
- cleaner future path to low-precision or neuromorphic-aligned export

Do not place audit-critical source truth only in binary payloads. Binary payloads should be paired with manifests.

## Recommended Package Shapes

### Research Dataset Package

Recommended shape:

- `manifest.json`
- `records.jsonl`
- optional `rejections.jsonl`
- optional `lineage.jsonl`

Use when the main object is a source-aware dataset slice or evaluation fixture package.

### Runtime Model Package

Recommended shape:

- `manifest.json`
- `state.msgpack`
- optional `trace_profile.json`
- optional `export_caps.json`

Use when the main object is a loadable inference or memory artifact.

### Edge Deployment Package

Recommended shape:

- `edge_model.json` or current edge export payload
- embedded or adjacent `edge_manifest`
- optional binary payload support in a future format revision

The current edge JSON format is acceptable for now because it prioritizes portability and strict validation. If payload size becomes limiting, the next step should be a manifest-preserving hybrid format, not an opaque export blob.

## Format Rules By Artifact Type

### Keep as JSONL

Keep these primarily line-oriented:

- `data/processed/autobot/*.jsonl`
- `data/interim/autobot/*.jsonl`
- synchronized-experience event and relation rows
- benchmark fixture rows
- operational repair logs and event traces

### Keep as JSON

Keep these manifest-oriented:

- benchmark reports
- readiness reports
- roadmap-support reports
- release action manifests
- session plans
- edge manifests

### Prefer Binary + Manifest

Prefer moving these toward paired binary payload plus manifest:

- final inference memory under `models/`
- sparse runtime caches reused across runs
- compact edge deployment state when current JSON export becomes too large
- any repeated numeric state snapshot whose rows are not normally audited by humans

## Migration Principle

Do not perform a repository-wide format rewrite in one step.

Migration should follow this order:

1. define canonical manifest fields
2. wrap existing payloads with those manifests
3. add read/write compatibility for both old and new payload shapes
4. migrate high-cost runtime state before migrating audit-friendly JSONL data
5. keep tests that verify backward compatibility until one stable release cycle passes

## Canonical Manifest Requirements

New manifests should converge on these common fields where applicable:

- `schema`
- `format_version`
- `artifact_type`
- `created_at`
- `source_artifacts`
- `record_count`
- `payload_path` or embedded payload indicator
- `payload_digest`
- `digest_algorithm`
- `lineage_policy`
- `compatibility_flags`

Not every manifest needs every field, but the direction should be toward convergence rather than ad hoc structure.

## What Not To Do

- Do not force all model state into `JSONL`.
- Do not force all datasets into one binary blob.
- Do not place provenance only inside runtime payloads.
- Do not require a heavyweight database as the default persistence surface for managed artifacts.
- Do not create hidden storage formats that bypass `project_paths.py`, policy, or manifest validation.

## Near-Term Recommendation

The best next bounded step is:

1. keep current `JSONL` dataset/event storage
2. keep current JSON reports and manifests
3. standardize model/runtime artifacts around `manifest + msgpack payload`
4. delay edge-format binary optimization until payload size or load time becomes a measured bottleneck

That path improves long-term structure without disrupting current Phase 6, Phase 7, and Phase 8 evidence work.

