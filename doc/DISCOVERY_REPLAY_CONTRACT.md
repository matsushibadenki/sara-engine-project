# Discovery Replay Contract v1

This is an observed-only research-orchestration primitive inspired by [Dream-RSI](https://github.com/zhengkid/Dream-RSI/blob/main/papers/Dream-RSI.pdf). It does not change SARA's neural learning, Event Memory replay, evaluator, or production behavior.

## Record and visibility

`ReplayRecord` is a frozen, versioned tree-node shape. Each record carries a contiguous sequence number, opaque node and parent IDs, hypothesis and policy IDs, SHA-256 candidate/source/preregistration identities, one fixed evaluator ID, a split, outcome status/score, and nonnegative CPU/event/state costs. Valid and negative-gate outcomes retain finite measured scores; execution failures and explicitly missing outcomes carry no score. Missing means a recorded action has no usable outcome; it is not a negative result. The root has no parent, result, or cost. Non-root records have an earlier parent; only the root may start multiple branches, and each non-root branch has at most one recorded continuation. The implementation accepts development trees only.

`DiscoveryReplayWorld.view` returns only revealed records. A policy should receive this view, not the world object or the complete tree. Selecting the root reveals the next recorded independent branch; selecting a visible branch tip reveals its one recorded continuation. Duplicate, invisible, terminal, unsupported, oversized, or over-budget selections fail before any state changes. An empty selection stops the rollout. No node outcome is synthesized for an unrecorded action.

## Bounds and exclusions

Defaults are at most 1,024 recorded nodes, eight actions per decision, and 128 revealed nodes. Callers may set smaller positive limits. All records in a tree must use the same evaluator ID; `heldout` records are rejected. `run_replay_policy` passes only `ReplayView` to a policy and caps decision rounds. It neither schedules experiments nor compares policies. No replay result authorizes changing a frozen benchmark, launching an experiment, or deploying a policy.

## Managed journal

`DiscoveryJournal` accepts only `.jsonl` paths under `workspace/`. Each canonical line contains exactly one v1 record, the preceding line hash, and its own SHA-256 digest. Append operations take an exclusive file lock, validate the complete prior prefix and tree, enforce a 1,024-record/4 MiB journal limit, then append and sync. A batch is fully validated before the write, but its disk append is not crash-atomic. Invalid, incomplete, noncanonical, duplicate-key, oversized, or held-out content fails closed. A returned head hash can be pinned elsewhere; `load` and `append` optionally require that expected head, detecting a stale or wholesale rewritten chain. A SHA chain without an independently pinned head is **not** authentication against a malicious full rewrite. An interrupted short write may leave a partial last line; it is rejected, not silently repaired.

## Reviewed export and post-run audit

`ExportReviewReceipt` binds a declared reviewer, source digest, and digest of one sanitized development tree. `export_reviewed_tree` accepts that exact tree only when review, raw-text-removal, and held-out-exclusion flags are true; it writes into an empty managed journal under a genesis-head compare-and-swap. This is a programmatic approval gate, **not proof of human review**: receipts are neither signed nor independently authenticated. No existing experiment histories have been reviewed or imported by this implementation. A human must inspect source provenance, redaction, split isolation, and the opaque IDs before issuing a receipt, then independently pin the resulting head hash.

`audit_replay` is an operator-only post-run inspection, never policy input. It reports recorded versus revealed non-root nodes, coverage, unrevealed nodes, missing and failed outcomes, and a lower bound on nodes structurally unreachable under the reveal budget. The lower bound assumes the shortest valid reveal path and is not a counterfactual estimate of what an untried policy would discover. The controller passes a `ReplayView` containing only currently revealed records, and hidden-node actions fail. That interface-level check does **not** prevent a Python policy closure or external process from reading the source tree out of band. Run policies in an isolated environment and audit their inputs before making unavailable-information claims.

`run_traced_replay_policy` starts from a fresh world and records a digest of the exact view supplied at each decision, the chosen actions, newly revealed node IDs, and the resulting view digest. `verify_replay_transcript` reconstructs each transition from the bound tree and rejects hidden actions, forged reveals, or mismatched views. A transcript is a reproducibility and interface-availability check, not proof that the policy did not read hidden data through a closure, filesystem, network, or another process. No isolated execution boundary is implemented yet.

The strict record allowlist excludes free-text evidence and source content, but opaque IDs can still encode sensitive information. Add task-family/time/source split isolation, independent source-redaction review, isolated-policy leakage checks, and prospective validation before any policy optimization claim.

日本語: この再生は研究の探索順を検討する観測専用機能です。未記録の結果は生成せず、held-outデータや本番推論には接続しません。

简体中文：此回放仅用于研究探索顺序的观察性分析，不生成未记录的结果，也不连接留出数据或生产推理。
