# Independent Data Collection Plan

**更新日 / Updated / 更新日期:** 2026-07-21
**対象 / Scope / 范围:** Phase 7, Phase 8, Architecture Migration, Phase 19/20, RISA  
**対象外 / Out of scope / 不包含:** Phase 6 external power measurements

この文書は、今後必要な独立データの収集先と収集方法を定義する。独立データが不足している場合は、ゲートを通すためにデータを捏造せず、`[Next]` の収集要求として保持する。

This document defines where and how to collect the independent data required by the roadmap. Missing evidence must remain blocked; fixture data, self-generated data, and proxy values must never be promoted as independent evidence.

本文档规定路线图所需独立数据的来源和采集方法。证据不足时必须保持阻塞状态，不得将fixture、自生成数据或proxy数据提升为独立证据。

## Status Summary

- [Done] Phase 7 already has 24 train and 24 evaluation rows across six disjoint source domains.
- [Done] Phase 7 isolation audit, block policy, request-level repair evidence, and rerun templates are implemented.
- [Done] Phase 8 required CPU comparison surface and Event Memory compression-maintenance coupling surface are implemented as observed-only evidence.
- [Done] Architecture Migration has a provenance gate, manifest builder, evidence cycle, and managed collection request.
- [Done] Collect independent long-horizon Architecture Migration records and pass the external gate.
- [Done] Collect independent UD English, Japanese, and Simplified Chinese train/dev/test material with split isolation audit.
- [Next] Collect additional independent raw-text role-binding evidence for Phase 19/20 promotion review.
- [Next] Collect the Phase 23 five-case multimodal minimum across at least two independent source domains.
- [Done] Phase 23 independent provenance, leakage, decision coverage, structural verification, and Event Memory admission gates are implemented.
- [Next] Keep raw-text promotion blocked until endpoint coverage, structural disambiguation, and bounded-state gates pass together.
- [Later] Collect independent structural evidence for RISA structural interpolation and smoothing.
- [Done] A local pretrained embedding reference is now available for the current Phase 8 comparison; FAISS and cross-encoder remain optional extensions.
- [Later] Provision optional local FAISS and cross-encoder references when their local dependencies are intentionally available.

## Common Collection Contract

Every independent row must preserve the following fields before preprocessing:

```json
{
  "schema": "sara-independent-source-row-v1",
  "record_id": "stable-record-id",
  "content": "source excerpt or observed event description",
  "source_url": "https://authoritative.example/document",
  "source_domain": "authoritative.example",
  "source_hash": "sha256-of-source-or-canonical-excerpt",
  "source_revision": "version-or-publication-date",
  "collection_time": "2026-07-17T00:00:00Z",
  "evidence_scope": "independent_external",
  "observed_only": true,
  "compliance_level": "allow",
  "license_hint": "license-or-permission-reference",
  "near_duplicate_signature": "16-hex-digit-signature",
  "task_type": "qa|negative|contrastive|delayed|migration|structural"
}
```

Required rules:

- Use primary or authoritative sources where possible: standards bodies, official project documentation, official manuals, release notes, and permissively licensed public datasets.
- Record the exact URL, revision or publication date, license information, and collection timestamp.
- Hash the canonical source excerpt before paraphrasing or deriving a task.
- Keep raw source records under `data/raw/` and cleaned records under `data/processed/`; reports and manifests belong under `workspace/`.
- Split before generating questions, summaries, contrastive pairs, or repair material.
- Do not use repository fixtures, model-generated text, or an answer generated from the evaluation set as independent evidence.

## Collection Targets

### Phase 7 Source-Aware Learning Data

**Purpose:** strengthen autonomous learning material while preserving train/evaluation isolation.

**Existing source pattern:**

- Train: `docs.python.org`, `www.w3.org`, `datatracker.ietf.org`
- Evaluation: `developer.mozilla.org`, `man7.org`, `docs.openstack.org`

**Future collection destinations:**

- Official programming-language and operating-system documentation
- Standards and protocol organizations
- Official project manuals and versioned release notes
- Public documentation with an explicit license and stable revision history

**Collection method:**

1. Register source domains and licenses in a managed collection target.
2. Collect source excerpts with URL, revision, timestamp, hash, domain, and license metadata.
3. Split by source document, source revision/hash, domain, collection time, and near-duplicate signature.
4. Create positive QA, negative query, unsupported query, contrastive, noisy, and delayed-recall rows only after the split.
5. Run the isolation audit before gap-material generation.

**Acceptance:** all five isolation axes pass, every row uses `evidence_scope=independent_external`, and the gap-loop completion gate does not depend on fixture-only rows.

### Architecture Migration

**Purpose:** verify that verified Event Memory and RISA knowledge survive an architecture-version change without replaying incompatible or unverified state.

**Required destinations:**

- At least two independent, non-placeholder HTTPS source sites
- Prefer versioned technical documentation, standards, release histories, or long-form manuals
- Do not use `example.org`, repository fixtures, or one source copied across multiple URLs

**Required volume and structure:**

- At least 3 records per independent source site
- At least 2 source sites before promotion review
- Unique material hashes for every record
- Long-horizon records with ordered `horizon_index` or equivalent sequence position
- Observed-only, compliance-allowed, source-backed records

**Collection method:**

1. Store raw excerpts in a managed JSONL input file.
2. Build a deterministic migration manifest with normalized domains and provenance digests.
3. Run `eval-architecture-migration-external`.
4. If blocked, run `build-architecture-migration-collection-request` and collect only the missing requirements.
5. Re-run `eval-architecture-migration-evidence-cycle`.

**Promotion rule:** target replay recall, source independence, horizon coverage, provenance, and migration cost must all pass. A blocked gate is the correct result when the required sources are unavailable.

### Phase 8 Optional ANN References

**Purpose:** add stronger offline CPU baselines without changing the SARA runtime.

**Collection destinations:**

- Local, operator-provided model directories for sentence embeddings
- Local FAISS CPU installation and the same embedding model
- Local tiny Transformer or cross-encoder model directory

**Collection method:**

- Do not download models automatically as part of the benchmark.
- Record model identity, revision, license, parameter count, embedding dimension, tokenizer identity, quantization, and CPU thread count.
- Run all references on the same corpus, query set, candidates, quality criteria, and measurement boundary.
- Keep model preparation and index construction outside the SARA production runtime.

**Current local reference:** `nomic-ai/nomic-embed-text-v1` is evaluated from an existing operator-managed cache as an `offline_reference`. The cache path is intentionally not copied into repository artifacts; reruns must record the resolved model revision and local path in the managed comparison report.

**Acceptance:** label every result as `offline_reference`; report quality, abstention, latency, memory, and event/cost proxy together; do not promote a stronger baseline from configuration metadata alone.

### Phase 19/20 Independent Language Evidence

**Purpose:** decide whether adaptive temporal dynamics or Semantic Echo mechanisms should be promoted beyond observed-only local fixtures.

**Collection destinations:**

- Licensed technical prose and documentation with stable revisions
- Publicly licensed dialogue or narrative material when speaker and series boundaries are available
- Operator-supplied English, Japanese, and Simplified Chinese evaluation material with explicit rights

**Required task families:**

- Long-distance dependency and role binding
- Negation, scope, anaphora, omission, and word-order variation
- Noisy and adversarial near-miss text
- Unsupported queries and abstention
- Delayed recall and continual revision
- Phonological or boundary ambiguity only when the source labels the relevant signal

**Collection method:**

- Split by document, author/speaker, series, source hash, and collection period before deriving tasks.
- Keep text-only, timestamp-shuffled, single-decay, and fixed multi-timescale controls.
- Disable external parser/LLM assistance during evaluation.
- Record language, source revision, task family, expected answer or abstention, and provenance for every case.

**Acceptance:** independent held-out quality improves over fixed SNN controls or matches quality at lower bounded cost, with no regression in abstention, state, event, or source-isolation checks.

### RISA Structural Interpolation

**Purpose:** evaluate structural-space precision improvement without treating graph proposals as durable truth.

**Collection destinations:**

- Independent source documents describing overlapping concepts from different domains
- Versioned documents containing revisions or explicit corrections
- Source-backed positive, negative, contrastive, and contradiction examples

**Required records:**

- Concept nodes and typed relations such as `instance_of`, `predicts`, `precedes`, and `contradicts`
- Source hash, revision, domain, acquisition time, context, confidence, and evidence type
- Subgraphs that can be aligned without sharing the same source document
- Unsupported-neighbor controls where a structurally similar concept must not receive an unverified edge

**Collection method:**

1. Build source-backed relation candidates without structural smoothing.
2. Freeze train/evaluation subgraphs before proposing merges or edits.
3. Compare graph merge, confidence interpolation, bounded structural edits, and structure-only controls.
4. Send every proposed edge through Concept Review, contradiction checks, and Event Memory admission.
5. Keep optional embeddings as an ablation and never as the durable truth path.

**Acceptance:** held-out structural utility or abstention improves without increasing durable false links, rewrites, state, maintenance cost, or isolation risk beyond declared ceilings.

### Delayed-Label Video And Multimodal Concept Discovery

**Purpose:** test whether SARA/RISA can form useful anonymous concepts from observations first and attach a sparse human label later, without leaking labels into representation learning or promoting unsupported structure.

**Collection destinations:**

- Publicly licensed video or audio-action datasets with recording/session, speaker/creator, and source boundaries.
- Operator-supplied recordings collected with consent and an explicit retention and usage policy.
- Small, source-backed multimodal clips with stable revisions and permission to retain derived event metadata.

**Required records:**

- Raw media reference, source URL or operator record ID, license/consent status, source hash, revision, collection time, and recording/session identity.
- Sparse observations for motion, trajectory, contact, sound, color, shape, object persistence, and change-point events; each observation must retain modality and timestamp provenance.
- Anonymous cluster ID and representative-event manifest created before semantic naming.
- Optional human label with annotator, timestamp, label scope, confidence, alternative labels, and “unresolved” or “mixed” outcome support.

**Collection method:**

1. Register sources and rights before retaining media or derived records; do not default to broad web scraping.
2. Split by recording session, speaker/creator, series, source hash, and collection period before extracting clips, events, clusters, or representatives.
3. Extract bounded sparse events and cluster them without label tokens or human action names in the input path.
4. Show a small, provenance-preserving representative sample to a human for delayed naming; keep the label as a semantic binding record rather than rewriting the anonymous cluster evidence.
5. Re-run Concept Review, contradiction checks, and Event Memory admission after naming. Keep unsupported, unstable, or mixed clusters provisional.

**Required comparison:** labels-first supervised control, self-supervised/late-label control, and RISA structure-only control must use the same source split and declared event/state/maintenance budgets.

**Acceptance:** held-out cluster purity, naming consistency, hierarchy quality, non-language transfer, and mixed-cluster abstention improve or remain competitive without source leakage, unsupported durable links, or unbounded labeling and maintenance cost.

### Phase 23 Independent Multimodal Gate

**Purpose:** replace fixture-only structural-fusion evidence with independently collected, source-auditable observations while preserving abstention and Event Memory admission boundaries.

**Minimum batch:**

- At least five cases across at least two independent recording sessions or source domains, with at least two cases per domain.
- Two aligned vision/audio cases expected to produce `verify_cross_modal_structure`.
- One missing-modality case expected to remain `provisional_missing_modality_prediction`.
- One contradictory claim case expected to produce `abstain_cross_modal_contradiction`.
- One delay greater than 32 ms expected to produce `abstain_temporal_misalignment`.

**Manifest location:** `data/processed/autobot/phase23_independent_multimodal_manifest.jsonl`

Each JSONL row must use this shape:

```json
{
  "schema": "sara-phase23-independent-multimodal-row-v1",
  "case_id": "session-a-impact-001",
  "source_ref": "operator:session-a:clip-001",
  "source_hash": "sha256-of-canonical-clip-or-observation-record",
  "source_revision": "recording-v1",
  "source_domain": "session-a",
  "collection_time": "2026-07-21T00:00:00Z",
  "license_hint": "operator-owned-or-license-reference",
  "near_duplicate_signature": "stable-perceptual-signature",
  "evidence_scope": "independent_external",
  "observed_only": true,
  "compliance_level": "allow",
  "expected_modalities": ["audio", "vision"],
  "expected_decision": "verify_cross_modal_structure",
  "evidence": [
    {
      "modality": "vision",
      "label": "contact_motion",
      "claim_key": "impact_event",
      "timestamp_ms": 10.0,
      "source_ref": "operator:session-a:clip-001:vision",
      "source_hash": "sha256-of-vision-track"
    },
    {
      "modality": "audio",
      "label": "impact_sound",
      "claim_key": "impact_event",
      "timestamp_ms": 18.0,
      "source_ref": "operator:session-a:clip-001:audio",
      "source_hash": "sha256-of-audio-track"
    }
  ]
}
```

**Collection method:**

1. Prefer operator-owned recordings with consent, or explicitly licensed retained media; record rights before extracting events.
2. Use separate recording sessions or unrelated licensed sources as domains. Do not split one clip into artificial domains.
3. Preserve modality-local timestamps and references. `claim_key` expresses the shared proposition while `label` remains modality-local.
4. Hash the canonical clip or immutable observation record, then compute a near-duplicate signature before selecting evaluation cases.
5. Do not reuse repository fixtures, generated media, or `fixture:`, `synthetic:`, or `generated:` references.
6. Run the external gate, then regenerate only missing targets if it remains blocked.

```bash
python scripts/sara_cli.py eval-phase23-external-multimodal
python scripts/sara_cli.py build-phase23-multimodal-collection-request
```

**Acceptance:** every provenance and coverage check passes, verifier decision accuracy is 1.0, and only aligned verified cases cross the Event Memory admission boundary.

## Verification Workflow

Run the following in order after each collection batch:

```bash
python scripts/sara_cli.py eval-phase7-isolation
python scripts/sara_cli.py apply-phase7-isolation-block-policy
python scripts/sara_cli.py eval-autobot-gap-loop-readiness
python scripts/sara_cli.py eval-phase7-completion
python scripts/sara_cli.py eval-architecture-migration-evidence-cycle
python scripts/sara_cli.py eval-phase8-evidence-cycle
python scripts/sara_cli.py eval-phase23-external-multimodal
python scripts/sara_cli.py build-phase23-multimodal-collection-request
python scripts/sara_cli.py eval-research-benchmark-suite
```

If any isolation, provenance, license, or quality gate fails:

- Keep the affected rows blocked.
- Preserve the report and failed-axis evidence under `workspace/evaluation/`.
- Generate a managed collection request instead of relaxing the threshold.
- Do not merge the affected rows into durable Event Memory or a promotion manifest.

## Data Handling Prohibitions

- No broad web scraping as the default collection strategy.
- No automatic network download of model weights during evaluation.
- No fixture-to-evaluation reuse.
- No source-domain overlap across train and evaluation for independent evidence.
- No removal of provenance fields to make an audit pass.
- No promotion based only on proxy energy, abstract event counts, or self-generated quality labels.

## Related Documents

- [ROADMAP.md](ROADMAP.md)
- [ARCHITECTURE_MIGRATION.md](ARCHITECTURE_MIGRATION.md)
- [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)
- [STORAGE_FORMAT_STRATEGY.md](STORAGE_FORMAT_STRATEGY.md)
- [policy.md](policy.md)
