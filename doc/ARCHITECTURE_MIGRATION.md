# Architecture Migration Protocol

## Goal

Preserve verified learning assets when SARA architecture changes without treating an old
route-weight layout as automatically compatible with a new circuit.

## Compatibility Boundary

- Do not directly copy sparse route weights across an incompatible neuron, routing, or
  plasticity-rule change.
- Preserve verified Event Memory as the canonical architecture-independent learning asset.
- Preserve source lineage, canonical concept keys, and causal predecessors with every
  replay candidate.
- Keep the legacy cache read-only during migration.
- Route incompatible, unverified, low-utility, or non-canonical entries to an explicit
  hold list for review instead of silently importing them.

## Migration Flow

1. Label legacy Event Memory with `architecture_version`.
2. Create a target cache for the new architecture version.
3. Build a bounded replay plan from verified, source-backed legacy entries.
4. Re-admit replay candidates into the target cache with both target and source
   architecture versions recorded.
5. Run normal idle replay, Concept Review, and RISA evidence coupling on the target
   cache before any route is considered stable.
6. Retain the legacy cache as a read-only reference until regression and source-isolation
   checks pass.

## Current Runtime Surface

- `EventStateCandidate` and `EventStateEntry` preserve architecture lineage.
- `ArchitectureMigrationCoordinator` builds a bounded, source-backed replay bridge.
- `IdleConsolidationLoop` can emit a migration plan or execute migration into an explicit
  target cache.
- `eval-architecture-migration` provides a frozen source-isolated baseline for recall,
  revalidation, RISA reconstruction, hold handling, and migration-cost checks.
- Older Event Memory snapshots remain readable and default to `sara-architecture-v1`.

## Promotion Rule

Migration success is not established by successful admission alone. Promotion requires a
frozen long-horizon comparison showing that the target preserves verified retrieval and
contradiction recovery within replay, state, and maintenance-cost caps, without source
isolation regressions.
