# SARA Engine Documentation Hub

This hub is the current entry point for the SARA Engine documents. It separates active operational documents, completed implementation history, future roadmap work, and archived research notes.

## Current Status

SARA Engine is a CPU-first, SNN-based research and runtime project. The active implementation follows these constraints:

- No runtime dependency on global gradient backpropagation; bounded backward information and hierarchical local credit are allowed.
- No dense-matrix-first design.
- No GPU requirement.
- Event-driven, sparse, biologically inspired mechanisms are preferred.
- Generated reports and temporary artifacts stay under managed paths: `workspace/`, `data/`, and `models/`.

The v1.1 release surface is completion-gated. The current gates validate Phase 3/4/5 completion, strict operational readiness, real-data external validity, sparse diffusion block readiness, ANN-efficiency roadmap evidence, and the research-product completion surface.

## Active Documents

- [Policy](policy.md): design and output-location rules that new implementation should follow.
- [Roadmap](ROADMAP.md): post-v1.1 development direction and next research/product phases.
- [Next-Level Roadmap](ROADMAP_NEXT_LEVEL.md): Phase 21-28 goals for structural cognition, continual horizons, multimodal world modeling, and verifiable agency.
- [Implemented Features](IMPLEMENTED_FEATURES.md): completed v1.1 feature inventory and gate status.
- [Architecture Review](ARCHITECTURE_REVIEW.md): design-health review, architecture spine, research adoption rules, and risk controls.
- [Release Checklist](RELEASE_CHECKLIST.md): commands and review steps for release, operational readiness, and v1.1 gates.
- [Release Notes](RELEASE_NOTES.md): current release-candidate summary.
- [Tools](TOOLS.md): active CLI, evaluation, release, and maintenance command map.
- [Training Manual](SARA-Engine_Training_Manual.md): data import, training, inference, memory inspection, and cleanup guide.
- [Competitive Analysis](COMPETITIVE_ANALYSIS.md): comparison against existing SNN libraries and remaining strategic gaps.
- [Storage Format Strategy](STORAGE_FORMAT_STRATEGY.md): long-term division between append-only records, canonical manifests, and compact runtime payloads.

## Research And Idea Archive

Exploratory papers, older tool lists, diagrams, and research notes live under [doc/idea](idea) or [doc/old](old). These files are useful as design input, but they are not the canonical implementation or release procedure.

Use the archive for:

- Research intake and future architecture ideas.
- Legacy tool descriptions that no longer match the active script surface.
- External papers, PDFs, diagrams, and notes that have not been promoted to active acceptance criteria.

## Current Snapshot

- v1.1 release gate: `15/15` checks passing.
- Research product completion gate: `12/12` checks passing.
- ANN-efficiency roadmap gate: `6/6` stages passing.
- Full test suite: `904` tests passing in the Python 3.10 project environment.
- Current next focus: paired physical energy measurement, stronger external baselines, research-grade benchmark packaging, hardware portability profiles, and operator usability.

## Recommended Review Flow

1. Read [policy.md](policy.md) before implementation.
2. Check [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md) when deciding whether a new feature belongs in active implementation.
3. Check [IMPLEMENTED_FEATURES.md](IMPLEMENTED_FEATURES.md) before assuming a feature is missing.
4. Check [ROADMAP.md](ROADMAP.md) for the current post-v1.1 phase target.
5. Use [TOOLS.md](TOOLS.md) to find the correct command.
6. Run the relevant gate from [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
7. Update [RELEASE_NOTES.md](RELEASE_NOTES.md) when behavior, release gates, or operational procedures change.
