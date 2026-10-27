# Nonstationary candidate reactivation — 2026-10-26

🟢 [Done] Preregistered and implemented the separately authorized component comparison. **The registered component gate fails.** Primary route-update checks pass, but all ten changed cases fail recovery within 256 ticks. A full report replay was attempted and failed direct equality; no successful replay receipt exists. Preserve both failures without retuning or restarting the consumed attempt.

This work follows the owner's delegated choice between a new metadata audit and this separate component experiment. It does not reopen the R2 independent-task/resource hold, any real dataset or a sealed test period. It is a non-spiking local-memory experiment, not evidence of an ANN-absent SNN capability.

## Fixed protocol

The [selection](../data/processed/benchmark_fixtures/sparse_route_reactivation_selection_v1.json) was saved on **2026-10-26 at 09:58:31 JST**, before candidate implementation. [Code/tests registration](../data/processed/benchmark_fixtures/sparse_route_reactivation_registration_v1.json) at **10:01:03 JST** preceded execution. Five fresh study seeds `1201, 1213, 1229, 1237, 1249` are disjoint from prior pruning seeds and development-test seeds. Two tasks × five seeds × stationary/changed conditions give 20 cases, each with five arms and 1,024 chronological prediction/outcome receipts.

The generator supplies anonymous permutations of five binary inputs `(a,b,c,d,e)` and three output positions. Learning excludes core `111`; nuisance channels are IID. Initial targets are `(a,b,c)` or `(a,b XOR c,c)`. At tick 512, changed conditions use `(d,e,a)` or `(d XOR e,e XOR a,a)`; stationary conditions keep the original mapping. Inputs and initial prefixes match across conditions. Learners receive only current events and their one-tick delayed outcome, never latent variables, dependencies, condition or change time. The evaluator alone inspects dependency retention and recovery. Frozen evaluation uses 256 independent IID packets, including core `111`.

This is a new nonstationary generator and weighted residual rule; it is not a repeat of, or direct numerical extension to, the stationary pruning result. Prior input-overlap and R2 negative results remain unchanged.

| Arm | Fixed behavior after common 180-receipt warmup |
| --- | --- |
| `round_robin` | Keep 45 candidates; update three distinct routes per output uniformly. |
| `permanent` | Select three lowest-score routes per output once and retain those nine. |
| `periodic` | Start 180-receipt full-bank exploration at ticks 436, 692 and 948, retaining the old nine-route readout until each completed window commits. |
| `error_triggered` | Use delayed prequential Brier EWMA, rate 1/16 and threshold 0.20, to start the same bounded exploration. Commit a new nine-route pool only after 180 observed receipts, then wait 64 ticks. |
| `shuffled_error` | Same candidate with outcome packets independently permuted within each phase; training still consumes one delayed packet per prediction. |

Every arm performs **nine actual route updates per tick**, with three per output. Each route update changes one observed class count, its observation count, a positive weighted squared-residual sum and a positive weight sum. Public-time weight is `2**(tick//64)`; score is residual sum / weight sum plus `0.005*key width`. Smoothed pre-outcome probabilities keep residuals strictly positive in this finite run; every one of the four stored changes is checked against its previous value. There is no error/table reset, padded no-op, hidden learning pass or penalty for sharing useful inputs.

Each arm/case executes **9,216 route updates and 36,864 actual changes of these four learning fields**. All 100 arm/case runs match, with independent count-table and observation totals. This is not equality of every state write or operation: detector, scheduling, pool changes, table initialization and lifecycle metadata remain separate. Exploration starts and commits are retained; detector/scheduler actual changed-field counts and total service cost are not fully quantified. No total-work, memory, CPU or energy advantage is claimed. The fixed 45-route bank, binary key capacity of at most four per route, one pending receipt and bounded lifecycle metadata are algorithmic bounds, not deployment measurements. Retired tables remain in memory.

## Primary results and negative gates

The following percentages are five-seed means from the retained primary report. They are descriptive results of this exploratory toy study, **not fully replay-verified evidence**.

| Changed condition / arm | Independent tail exact | Independent frozen exact | Interacting tail exact | Interacting frozen exact |
| --- | ---: | ---: | ---: | ---: |
| Full-bank round-robin | 97.81% | 93.98% | 97.19% | 93.98% |
| Permanent pruning | 52.19% | 59.77% | 23.59% | 23.12% |
| Periodic exploration | 100% | 100% | 98.75% | 100% |
| Error-triggered exploration | 100% | 100% | 89.06% | 90.86% |
| Shuffled outcomes | 16.88% | 14.45% | 19.22% | 14.53% |

Tail exact uses the final 128 prequential predictions. Recovery requires two consecutive complete 64-tick post-change blocks at ≥90%, with the second block ending within 256 ticks. Error-triggered exploration starts at ticks 518–525. Nine cases reach the required pair only at **320 ticks**, and one interacting case never reaches it. **All ten changed cases fail the fixed recovery gate.** Thresholds, window length, seeds and the recovery definition are not changed.

Three changed cases also fail exact true-dependency retention, two fail the required ≥10-point tail gain over permanent pruning, and one interacting case fails both tail and frozen ≥90% accuracy. That case has tail 45.31% and frozen 54.30%; averaging must not hide it. The initial-new-dependency-absence gate and outcome-alignment gate pass in all ten cases. Exact dependency retention is a conservative registered criterion; accurate supersets do not retroactively rescue its failures.

All ten stationary cases pass: error-triggered exploration starts **zero** unnecessary windows and matches permanent pruning at 100% tail/frozen accuracy. Periodic exploration starts three windows per stationary case. Its first changed-condition window already began at tick 436, before the tick-512 change; its recovery cannot be attributed to faster change detection or a spike-specific benefit.

## Replay failure and verification boundary

The primary [result](../data/processed/benchmark_fixtures/sparse_route_reactivation_result_v1.json) is 464,131 bytes. Its `execution_integrity_passed=true` covers the implemented learning-count and within-case frozen scalar/order/replay checks. It does **not** certify the later whole-report replay. The fresh-process replay consumed its own immutable marker and exited with `ValueError: Exact deterministic replay differs`. The [failure receipt](../data/processed/benchmark_fixtures/sparse_route_reactivation_failure_v1.json) records `full_replay_verified=false` and numerical trace equality as **unknown**; there is no successful replay receipt or replacement run.

A known comparator defect is that `initial_pool` is a tuple in the returned Python report and a list after JSON serialization. Direct Python equality rejects these representations even when their elements match. Post-result regression checks reproduce that representation mismatch from saved metadata. They do not prove it is the only mismatch, or establish equality of the complete discarded replay report. The registered source is left untouched; the consumed primary/replay cannot restart. A future codec check must use saved reports or symbolic schema fixtures first, without rerunning learners or treating shape normalization as numerical replication.

**121 tests pass under Python 3.10.22**: 17 preregistered development mechanics tests, four explicitly post-result failure regressions, and 100 retained contract/decision/path tests. They cover actual four-field changes, inactive-route preservation, common warmup, delayed outcome/atomic rejection, exploration boundaries, cooldown, fixed periodic timing, hidden change information, source/evidence pins and preserved failures. No old experiments or resource probes are rerun by this test selection. All **399 pinned prior artifacts** and the registered sources verify.

| Artifact | SHA-256 |
| --- | --- |
| Selection | `4270ed450df82d391cdefa2204e207f1e1a3c7d66fb8b50fe5b053db1b789f4a` |
| Code registration | `72447318c97fbe22fe03412ba6351236b496e413369f5fd8469bebf6d14d1486` |
| Primary result | `66e75f708314e771d22d938239532e161f230c9fd334fe7fa4127822681b3c07` |
| Consumed replay marker | `b674f53ac162cbb48cdc369f4525f13ae8b89bd225fba584f07710fe0363f2b2` |
| Failure receipt | `d07ff2b2169a319b70b06be579d66eab6c59fafd20f83a8fa3d230b62a5875ba` |

Implementation: [learner/evaluator](../src/sara_engine/evaluation/sparse_route_reactivation.py), [one-shot runner](../scripts/eval/sparse_route_reactivation.py), [development tests](../tests/test_sparse_route_reactivation.py), [post-result failure checks](../tests/test_sparse_route_reactivation_failure.py).

```sh
# Read-only verification tests; do not restart execute or replay.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_sparse_route_reactivation.py tests/test_sparse_route_reactivation_failure.py tests/test_local_credit_contract.py tests/test_r2_mechanism_decision.py tests/test_r2_research_decision.py tests/test_project_paths.py
```

🟢 [Done] [Saved-report serialization/equality preflight](SPARSE_ROUTE_REACTIVATION_CODEC.md): registered before implementation, parses only the saved primary/failure reports (465,308 bytes). All 19 checks and 35 tests pass; exact static replay preserves 412 hashes. Tuple/list interchange is restricted to 80 initial-pool paths; numeric/type/order/field differences are not hidden. Zero learner execution/scoring/acquisition/measurement. Original replay failure and negative gates remain unchanged; full numerical equality stays unknown.
🟢 [Done] [Fixed-condition numerical reproduction v2](SPARSE_ROUTE_NUMERIC_V2.md): separately registered new primary and fresh-process replication, unchanged learner/conditions/codec. All numerical payloads and registered historical-reference comparisons match exactly (only two explicit provenance-field bindings). Each phase runs 20 cases × five arms with matched 9,216 route updates / 36,864 changed learning fields per arm/case. All ten changed recovery failures remain negative. 56 tests and read-only proof verification pass; 422 prior pins and old failed replay/seals preserved. No SNN/resource promotion.
🟢 [Done] [Saved-report recovery/retention failure audit](SPARSE_ROUTE_FAILURE_AUDIT.md): preregistered seven inputs / 1,593,723 bytes, all 100 pool lifecycles reconcile. Trigger lag 6–13 and commit lag 186–193 ticks leave only 63–70 ticks to the fixed deadline; no claim that the original gate was impossible. Three exact-retention failures split into two high-accuracy cases and one poor-accuracy case; missing true-route identities/ranks/support and counterfactual effects remain unknown. 90 tests, exact pure replay and 432 prior pins; zero learner/scoring/acquisition/measurement/condition changes. Old failed replay, negatives and seals preserved.
🟢 [Done] [Prospective bounded route snapshot contract](SPARSE_ROUTE_SNAPSHOT_CONTRACT.md): external read-only all-45-candidate collector, pre/post selection invariants, recent-window count/error differences and evaluator-only dependency diagnostics implemented. Fixed saved-metadata plan: 160 future hooks; conditional encoded bound 1,376,417 bytes <2 MiB. Static preflight/replay and 127 tests pass, preserving 441 prior pins. Actual collection/learner/scoring remains zero; no causal attribution, SNN advantage or trial authorization.
🟢 [Done] [Fixed-condition real route capture v3](SPARSE_ROUTE_CAPTURE_V3.md): separately preregistered baseline/capture executions, unchanged science/collector. All numerical payloads match; 160 actual boundaries and 160 full-state read-only checks pass. Missing dependencies have 72/412/72 observations and commit ranks 4/8/4: score exclusions, not ties or original 16-observation failure. Window re-ranking does not recover them; causal/counterfactual conclusions remain unknown. 143 tests, 450 prior pins, original negatives/replay failure/seals preserved; no promotion/resource claim.
⭕️ [Pending] Further mechanism changes require separately justified causal/task evidence and a new protocol. No automatic tuning, old-attempt restart or additional unregistered run. Completed v3 evidence collection does not reopen nonstationary promotion or the authoritative R2 independent-task/resource holds.

⭕️ [Pending] Nonstationary mechanism promotion and further predictive experiments remain held after the recovery/retention failures. R2 fresh independent task qualification, causal delayed-credit evidence, physical energy and production remain under their existing holds. Circuit-selective reactivation remains a shared non-spiking design opportunity, not an established SNN advantage.

日本語: 別枠の非定常比較を実装前に登録し、20ケース・各5方式を実行しました。全方式で9,216回の局所更新と36,864個の学習フィールド実変更が一致しましたが、変更10ケースすべてで256tick以内の回復に失敗し、部品基準は不合格です。定常条件の不要な再探索は0回でした。全報告の再現比較も失敗し、tuple/listの型差を確認しましたが、完全な数値一致は未確認です。121テストと399既存証跡の保持を確認し、再実行・調整・封印解除は行いません。

简体中文：在实现前注册独立的非平稳比较，执行20个案例、每例5种方法。各方法均实际执行9,216次局部更新、36,864次学习字段变化，但全部10个变化案例均未在256 tick内恢复，组件门槛失败。稳定条件下误触发探索为零。完整报告重放比较也失败，已确认tuple/list类型差异，但完整数值一致性仍未知。121项测试通过并保留399项既有证据，不重跑、不调参、不解除封存。
