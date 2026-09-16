# Discovery Replay Contract v1

This is an observed-only research-orchestration primitive inspired by [Dream-RSI](https://github.com/zhengkid/Dream-RSI/blob/main/papers/Dream-RSI.pdf). It does not change SARA's neural learning, Event Memory replay, evaluator, or production behavior.

## Record and visibility

`ReplayRecord` is a frozen, versioned tree-node shape. Each record carries a contiguous sequence number, opaque node and parent IDs, hypothesis and policy IDs, SHA-256 candidate/source/preregistration identities, one fixed evaluator ID, a split, outcome status/score, and nonnegative CPU/event/state costs. Valid and negative-gate outcomes retain finite measured scores; execution failures carry no score. The root has no parent, result, or cost. Non-root records have an earlier parent; only the root may start multiple branches, and each non-root branch has at most one recorded continuation. The implementation accepts development trees only.

`DiscoveryReplayWorld.view` returns only revealed records. A policy should receive this view, not the world object or the complete tree. Selecting the root reveals the next recorded independent branch; selecting a visible branch tip reveals its one recorded continuation. Duplicate, invisible, terminal, unsupported, oversized, or over-budget selections fail before any state changes. An empty selection stops the rollout. No node outcome is synthesized for an unrecorded action.

## Bounds and exclusions

Defaults are at most 1,024 recorded nodes, eight actions per decision, and 128 revealed nodes. Callers may set smaller positive limits. All records in a tree must use the same evaluator ID; `heldout` records are rejected. `run_replay_policy` passes only `ReplayView` to a policy and caps decision rounds. It neither schedules experiments nor compares policies. No replay result authorizes changing a frozen benchmark, launching an experiment, or deploying a policy.

## Managed journal

`DiscoveryJournal` accepts only `.jsonl` paths under `workspace/`. Each canonical line contains exactly one v1 record, the preceding line hash, and its own SHA-256 digest. Append operations take an exclusive file lock, validate the complete prior prefix and tree, enforce a 1,024-record/4 MiB journal limit, then append and sync one line. Invalid, incomplete, noncanonical, duplicate-key, oversized, or held-out content fails closed. A returned head hash can be pinned elsewhere; `load` and `append` optionally require that expected head, detecting a stale or wholesale rewritten chain. A SHA chain without an independently pinned head is **not** authentication against a malicious full rewrite. An interrupted short write may leave a partial last line; it is rejected, not silently repaired.

The strict record allowlist excludes free-text evidence and source content, but opaque IDs can still encode sensitive information. A separate source-redaction review and approved export step are required before real experiment histories enter a journal. Add task-family/time/source split isolation, recorded-action coverage, unavailable-information leakage checks, and prospective validation before any policy optimization claim.

日本語: この再生は研究の探索順を検討する観測専用機能です。未記録の結果は生成せず、held-outデータや本番推論には接続しません。

简体中文：此回放仅用于研究探索顺序的观察性分析，不生成未记录的结果，也不连接留出数据或生产推理。
