# Selected roadmap execution-gate consistency check — 2026-11-06

🟢 [Done] Corrected the stale discovery-roadmap item asking to preregister the fixed CartPole comparison again. That comparison was already preregistered, V1 failed serialization and V2 failed its frozen quality gate. The completed item now links to the retained V2 result rather than implying an unconsumed trial.

Added a four-gate register to both roadmaps and a read-only checker:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/check_roadmap_gates.py
```

The manifest is `data/processed/benchmark_fixtures/roadmap_gate_status_v1.json`. It checks unique gate markers and exact status/emoji in both roadmaps; hash-binds three authoritative documentation files; rejects the known consumed-work Next phrase; and refuses manifest runtime authorization. A 1 MiB aggregate input limit covers the two existing large roadmaps and the selected evidence documents. It writes no result artifacts and imports no model, learner, calculator, environment or persistence adapter.

| Gate | Current status | Basis |
|---|---|---|
| Fixed CartPole comparison | 🟢 [Done] | Comparison consumed; V2 negative and both held-out periods remain sealed. Done means implemented/completed, not successful quality. |
| Fresh local-credit task | ⭕️ [Pending] | New qualified task/hypothesis and fresh development identities are required; no retry or tuning of V2. |
| Actual receipt capture | ⭕️ [Pending] | Trusted owner/revision/pause and pre-outcome receipt access are still missing. |
| SNN-specific advantage | ⭕️ [Pending] | Synthetic compact timing equals refractory control; causal SNN superiority remains unestablished. |

Seven regression tests cover current consistency/nonmutation, stale Next wording, Pending-to-Next drift, missing/duplicate markers (including same-line duplicates), evidence changes, attempted authorization and roster/budget refusal. The first test run caught an undersized input cap; the documentation check was corrected to accommodate the measured existing document sizes, then all seven tests and the CLI passed. This is routine documentation-tool testing, not an experimental protocol adjustment or scientific rerun. No scientific threshold, task, input, source identity, old negative or seal changed.

This check is deliberately scoped to four selected gates. It is not a natural-language interpreter, global roadmap audit, security boundary or execution scheduler. Other stale wording can exist. A consistently edited manifest is not authenticated authority; updating an evidence pin requires reviewing the underlying outcome and prerequisites. No API from this checker authorizes a run. CPU/event counters in the retained CartPole reports remain proxies, not measured energy; local/event-driven behavior is also available to scalar approaches.

The current research priority is to resolve genuine integration/task prerequisites from independent source-backed evidence. Do not create another hand-fixture audit solely because the current line is blocked. Actual captures, numerical comparisons, storage transactions, acquisitions and resource measurements remain held under their existing gates. The broader roadmap's independent publisher pilot also requires an independently operated publisher and reviewed trust material; this change does not invent those prerequisites.

## 日本語

消費済みCartPole比較を再び事前登録するように見える古いNext項目を修正しました。4つの実行条件を両ROADMAPに同期し、7テストと読み取り専用CLIで確認しました。文書ハッシュ、状態表示、重複・欠落、既知の古いNext表現を検出します。これは計画の整合性チェックで、実験の許可や科学的成功の判定ではありません。新しい採点・取得・計測・学習は行わず、旧失敗・負の結果・封印を維持しています。

## 简体中文

修正了将已消耗的CartPole比较误列为待预注册的旧Next条目。在两份路线图中同步4项运行门槛，7项测试和只读CLI通过。检查文档哈希、状态标记、重复或缺失以及已知过期Next文字。这是计划一致性检查，不授权实验或判定科学成功。不新增评分、采集、测量或学习，保留旧失败、负面结果与封存。
