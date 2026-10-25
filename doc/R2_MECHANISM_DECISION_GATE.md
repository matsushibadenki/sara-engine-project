# Retained-evidence SNN mechanism and independent-task decision

🟢 [Done] A preregistered retrospective audit reconstructs the original predictive decision and reconciles completed accounting, separation and allocation receipts. It establishes no SNN predictive or deployment-resource advantage. Additional resource work is held; the synthetic measurement-feasibility branch is closed.

## Fixed protocol and implementation

The [selection](../data/processed/benchmark_fixtures/r2_mechanism_decision_selection_v1.json) was saved on 2026-10-23 at **20:37:55 JST**, before implementation. [Code registration](../data/processed/benchmark_fixtures/r2_mechanism_decision_registration_v1.json) at **20:49:30 JST** preceded execution. Outcomes were already known: this is a finite decision audit, not a blind confirmatory experiment.

Only 26 named aggregate JSON receipts are parsed, with a 600 KiB individual / 2 MiB combined parsing limit. Actual reads are **1,062,066 bytes**. Hash-only preservation checks cover **386 prior artifacts**, including original negative results, the failed allocation v1 and completed v2. Raw examples, model payloads and sealed periods are not semantically read. The new [pure evaluator](../src/sara_engine/evaluation/r2_mechanism_decision.py) reconstructs every field returned by the frozen original 19-receipt evaluator and compares it with its historical result. Ten learning rows retain their provenance; overlapping confidence supplements are not counted as independent experiments.

The [runner](../scripts/eval/r2_mechanism_decision.py) uses exclusive phase markers, fixed paths, bounded receipt reads and registered code/evidence hashes. Result output is limited to 256 KiB, with a 1 MiB total artifact budget. The actual result is **30,713 bytes**, and all six audit artifacts total **100,628 bytes**. A fresh-process read-only replay exactly reproduces the result. No experiment/probe worker, tracer, learner or scoring path is launched.

| Artifact | SHA-256 |
| --- | --- |
| Selection | `cfc43147a4cd801dbcccfb0fbc09bfd2541ec292e38d026c7385fe114bd11e20` |
| Code registration | `c472212d45c4e035511ab10a1d17599f60d11474d45445b8eb91a814452080d4` |
| [Result](../data/processed/benchmark_fixtures/r2_mechanism_decision_result_v1.json) | `9e3da66071bd2edefd01048468211fee78b639f0c9c8b2fc7ece6752653feaa8` |
| [Exact replay](../data/processed/benchmark_fixtures/r2_mechanism_decision_replay_v1.json) | `9889632b4e2ec0792e97045f267f6d5abef26997c575c30dd6f0a265966e88b0` |

## What the existing evidence permits

| Evidence | Retained interpretation |
| --- | --- |
| Controlled first-order history/local identification | Component success survives; sufficient scalar history does not establish an SNN-specific increment. |
| Joint middle-order/gap experiment | Joint superiority fails; its narrower reported gap-only positive result remains scoped to that comparison. No LIF retuning is justified here. |
| Room and independent household/appliance comparisons | Training-only identification is not independent usefulness. Independent negatives and quality failures remain negative; unreported frozen updates stay unknown rather than becoming zero. |
| Actual changed learning writes | Historical positive parity passes for the explicitly reported comparison arms, including sufficient confidence controls. This does not establish fairness for a new candidate. |
| Accounting/separation/allocation | Saved 48-case identities, state/receipt hashes and typed counters reconcile. Each standalone arm receives all 224 shared inputs; 1,280 calls per mode reconcile with 3,840 replicated windows. Counts of different work types are not summed into a cost score. |
| Allocation v1 / report preflight / v2 | Unsaved v1 measurements remain unavailable. Representation preflight is not measurement. Saved v2 neuron peak/graph samples are not smaller than scalar samples; CPU, physical energy, RSS and total process transient peak remain unknown. |

Event-driven scheduling and shared temporal observation are opportunities for both neuron and non-spiking implementations. The retained evidence does not isolate a spike-specific benefit. Diagnostic reproducibility cannot substitute for causal usefulness on an independent task.

## Conditional mechanism design and task prerequisites

The finite four-candidate roster retains shared temporal information, closes the known arrival-indexed LIF formulation, treats event-driven scheduling as shared, and selects **branch-local delayed eligibility** only as the next conditional design candidate. Its effect is **unknown**, not an observed success or a failed new experiment.

The causal question is whether bounded activity history on a branch can assign delayed outcome updates to the relevant past branch under identical observation, state and actual changed-write budgets. Routing and shared observation remain fixed. Updates must be local, sparse and auditable on CPU, without global differentiation, matrix dependence or GPU dependence. No structural growth, private extra history, label conversion or backdating may create the effect.

A separate future protocol would need strong exact-current/time controls, a sufficient-history non-spiking local readout, eligibility-disabled/permuted causal ablations, and an equation-matched scalar eligibility emulator. Equality with an emulator of the same equations is a correctness condition; superiority over that emulator is not claimed. All learning arms must perform the same legal positive number of actual changed writes, including changed counters. If parity requires dummy, no-op or duplicate updates, or changed eligibility conditions, the candidate is ineligible. Arrivals, delayed delivery, waiting, expiry and state limits must match. A causal ablation effect and independent usefulness must precede another resource protocol.

No new source is selected. The fixed exhausted CASAS list has no survivors; fixed public and SPHERE/ECO reviews authorize no payload acquisition, and the consumed HH105 arrival gate fails. These are finite-roster conclusions, not a claim that every possible source is impossible.

A future independent task must satisfy the unchanged original native **Kitchen ON** semantics: exact raw sensor/message plus weekday/hour/15-minute current state; eight history events, maximum 300-second gap, 60-second cutoff spacing and 300–600-second outcomes. Missing outcomes are not negatives. Minimum 180 recorded dates, the original fit/development split (**270 dates for 180 fit dates**), and **70,212 eligible arrivals** remain required. Exact-current recurrence must include sufficiently supported opposing histories, both outcome classes and recorded dates under the original support/effect rules. Unknown events remain included within acquisition/state budgets. All original quality, update, frozen-period and sealed-test gates remain in force. No source substitution, power/occupancy relabeling, threshold reduction or seal opening is authorized.

## Decision and verification

⭕️ [Pending] Additional resource work requires all five predicates to be evidenced: a qualified fresh independent task, independent usefulness, a causal mechanism effect with matched controls, positive actual-write/state fairness, and a decision-changing resource question. Retained independent comparisons do not pass usefulness; the fresh task and untested candidate predicates remain unknown. False and unknown are recorded separately, and either blocks further resource work. Even an all-true hypothetical unit fixture only permits consideration of a separately registered protocol; this audit never authorizes measurement.

No automatic v3, new acquisition/search, scoring, measurement, optimization or runtime promotion follows. Candidate/task prerequisite design may be revisited under a separate qualifying contract when new compatible evidence exists; this completed audit is not a recurring measurement or tuning loop.

**63 tests pass**, including mutated seals, original task conditions, roster/budgets, actual-write parity, source conclusions, shared charges, service reconciliation, replication, invented v1 statistics and unknown-to-zero resource values. The retained result and exact replay verify; zero new measurements, scores, acquisitions, learning writes, raw/model semantic reads and diagnostic workers are recorded.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/r2_mechanism_decision.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_r2_mechanism_decision.py tests/test_r2_research_decision.py tests/test_event_allocation_failure.py tests/test_project_paths.py
```

日本語: 26件の既存集計証跡だけを使う有限監査を実装前に登録し、元の予測判断と計測証跡を再構成しました。63テストと別プロセスでの完全再現が成功し、386既存証跡・負の結果・条件・封印を保持しました。枝ごとの遅延適格性を条件付きの設計候補に整理しましたが、その因果効果と新しい独立課題は未成立です。追加資源計測を保留し、新しい測定・採点・取得・条件緩和は行いません。

简体中文：在实现前注册仅使用26项既有汇总证据的有限审计，重建原有预测判断并核对计量证据。63项测试及独立进程完全重放通过，保留386项既有证据、负结果、原条件和封存。分支局部延迟资格迹仅作为有条件的设计候选，其因果效果与新的独立任务尚未成立。暂停额外资源测量，不开展新测量、评分、获取或条件放宽。

🟢 [Done] Follow-up prerequisite design: the [static local-credit proposal contract](R2_LOCAL_CREDIT_CONTRACT.md) checks causal metadata and proposed changed-field parity without running a learner. It does not change this decision or authorize a new experiment.
