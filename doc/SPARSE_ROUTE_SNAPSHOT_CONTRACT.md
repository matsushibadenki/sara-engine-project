# Prospective bounded route-evidence contract — 2026-10-27

🟢 [Done] Implemented an external, read-only snapshot collector and validator to address the missing route-level evidence identified by the [failure audit](SPARSE_ROUTE_FAILURE_AUDIT.md). The static contract/capacity preflight and exact pure replay pass. **No actual model snapshots have been collected, and no learner or new experiment has run.** Old states are not reconstructed; the existing causal unknowns, negatives, failed replay and seals remain unchanged.

## Registration and finite boundary plan

The [selection](../data/processed/benchmark_fixtures/route_snapshot_contract_selection_v1.json) was saved at **2026-10-27 19:47:44 JST**, before implementation. [Code/tests registration](../data/processed/benchmark_fixtures/route_snapshot_contract_registration_v1.json) at **19:56:24 JST** preceded the static execution and fresh-process pure replay. This is a prospective evidence-interface design informed by known retrospective results, not an unseen-data experiment.

Only two saved inputs are parsed: the numerical-v2 primary report and the completed failure-audit result, **595,549 bytes** combined. The existing starts/commits/final boundaries define **160 proposed future hooks** across all 20 stationary/changed cases for the error-triggered candidate and permanent control. This is a diagnostic subset of the old five-arm study, not a removal of its other controls or a change to its gates.

| Proposed boundary records | Error-triggered | Permanent | Total |
| --- | ---: | ---: | ---: |
| Before/after initial shortlist at tick 180 | 40 | 40 | 80 |
| Before/after exploration start | 20 | 0 | 20 |
| Before/after completed exploration commit | 20 | 0 | 20 |
| Final, after the last observed outcome | 20 | 20 | 40 |
| Total | **100** | **60** | **160** |

These positions are a plan derived from saved metadata, not evidence that old snapshots exist. Any authorized future execution must bind its source/registration and declared hook roster; if actual boundaries differ from its registered plan, it must stop rather than silently add records. Current code provides the contract, not an enabled learner hook adapter or authorization to collect.

## Collector, paired checks and evaluator separation

The [new module](../src/sara_engine/evaluation/route_snapshot_contract.py) accepts an explicit view of existing learner fields at a pending-free boundary. It calls no learner methods, predicts no output, and does not mutate tables, counts, weights, errors or scheduling state. All **45 candidates** are copied, including retired candidates; no route is clipped or only selected winners retained. The fixed input-slot/output catalog is stored once and is independent of seed permutations or latent truth.

Each compact route row contains its cumulative observation count, weighted prequential squared-error sum, public-time weight sum, and every canonical binary-key class-count cell. These are the original local-learning fields. The collector verifies class-count/observation equality, nine actual observations per elapsed tick globally, three per output, and conservation of the original public-time weight schedule. Finite numeric types, original binary capacities, route catalog, pending state, point identity and per-record size are checked before a record is released.

Snapshots also contain the active pool, a read-only calculation of the unchanged top-three-per-output ranking, and a digest of the supplied learner fields. Pre/post initial/commit pairs must preserve every learner row while the new pool matches that fixed ranking. Exploration-start pairs must preserve the active pool too. The collection encoder rejects missing, extra, duplicate, reordered or unpaired declared records, altered provenance and mixed symbolic/execution collections. It never truncates overflow.

Exploration-start and pre-commit endpoints can supply **recent-window** evidence by subtraction: per-route/per-key count increments and positive weighted-error/weight increments over the original 180 observed receipts. Every route must show the original 36 exploration observations; no reset or hidden learning pass is introduced. This permits future comparison of cumulative rankings with window statistics. It does not simulate the accuracy of a reset/windowed learner or show that recent support caused a failure.

True dependency IDs are supplied only to a separate offline evaluator API. They are never accepted by `capture`, inferred from old seeds or sent back to a learner. When absent, dependency analysis remains unknown. When explicitly supplied in a future authorized run, diagnostics can distinguish:

- Total readout support below the unchanged 16-observation requirement.
- A route excluded from the fixed top three by its score or deterministic ID tie-break.
- A route actually retained in the active pool.

Those are structural descriptions of supplied evidence, not causal explanations. Support shortage, score ranking and exact retention are not interchangeable; a low-support route can still rank in the shortlist. Functional substitution and the effect of changing a learning rule remain unknown. The validator cannot establish that caller-supplied state or evaluator IDs are truthful: a future executor must bind provenance and prove trace equivalence under its separate protocol. Digests are not a replacement for that evidence.

## Capacity and static verification

The per-record acceptance ceiling is **8,192 encoded bytes**. A complete header **including outer framing** is limited to **65,536 bytes**; at most 160 records are accepted. The actual-format upper bound is `160 × 8192 + 65536 + 159 = 1,376,415` bytes. The preregistered conservative bound adds two spare framing bytes, **1,376,417 bytes**, still below the unchanged **2 MiB** evidence-report ceiling. This is conditional on every record/header passing its byte gate. It is not a memory/CPU measurement, a guarantee about arbitrary objects, or permission to increase future total experimental storage budgets.

The [static runner](../scripts/eval/route_snapshot_contract.py) checks roster agreement with the saved report and exercises the API using explicitly labeled, mathematically consistent **symbolic metadata**. Its two-record fixture encodes to **5,172 bytes**. These records are not sampled from, or substituted for, any model. The retained static result is **1,278 bytes**; six static artifacts total **92,606 bytes**, within the 256 KiB individual / 1 MiB aggregate static ceilings. **441 prior file hashes** and registered sources verify.

**127 tests pass under Python 3.10.22**: 37 new collector/contract tests plus 90 retained pure audit/comparison/codec/path tests. They cover mutation and alias isolation, all-route completeness, invalid numeric/key/count/catalog fields, per-output conservation, pending/future boundary rejection, selection pair integrity, separate evaluator IDs, tie-breaking, low-support versus ranking, recent-window subtraction, collection ordering/provenance and byte overflow refusal. No learner or seed decoder is imported by this test selection. The static result exactly reproduces in a fresh process.

| Artifact | SHA-256 |
| --- | --- |
| Selection | `04ceb20739961386c075ecbbff6d110ae3a3eb50ce4fb8b28e5ffb028b803663` |
| Code registration | `3a45218cefab27355ce2a618d64503b1feb5c450049eed54c1aec1b54c9aaeb3` |
| [Static result](../data/processed/benchmark_fixtures/route_snapshot_contract_result_v1.json) | `71d7aea8d7127ad942212d25e0d531c911502e7412bb9dfe72368cef12b88e2d` |
| [Pure replay](../data/processed/benchmark_fixtures/route_snapshot_contract_replay_v1.json) | `37d7246acaab521a76275e84c6bcbe5549e40fb5c9da531f4b262124e485ebc4` |

```sh
# Contract verification only: no model state collection or learner execution.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/route_snapshot_contract.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_route_snapshot_contract.py tests/test_route_failure_audit.py tests/test_route_numeric_v2.py tests/test_reactivation_codec.py tests/test_project_paths.py
```

⭕️ [Pending] Actual hook integration and collection require a separately authorized future protocol with unchanged scientific conditions, comparison against the frozen uninstrumented traces and bounded source/evaluator provenance. Collection creates extra observer/encoding work; it must be accounted for and cannot be called free or an SNN energy benefit. No trial, old-attempt restart, new scoring, condition tuning, resource diagnostic or additional automatic audit is queued here. Existing R2 independent-task, causal branch-local credit, nonstationary promotion, energy and production holds remain in force.

日本語: 未保存だった候補別支持数・誤差・順位を次回の証跡に残すため、外部の読み取り専用採取器と検査器を事前登録・実装しました。全45候補、選択前後の不変性、直近180tickの差分支持数、評価者専用の依存ID分離を検証しました。計画は候補100／永久削減60の計160境界で、条件付き保存上限1,376,417バイトは2 MiB以内です。127テストと静的完全再現が成功し、441既存ハッシュを保持しました。実モデルの採取・学習器実行・採点はゼロで、原因の未知、旧再現失敗、負の結果、封印を維持します。実験への接続・実採取は別登録・承認が必要な保留事項です。

简体中文：为未来记录未保存的候选支持数、误差和排名，预注册并实现外部只读采集器与验证器。覆盖全部45个候选、选择前后不变性、最近180 tick的差分支持及评估者专用依赖ID隔离。计划160个边界记录，条件性上界1,376,417字节低于2 MiB。127项测试及静态完全重放通过，保留441个既有文件哈希。实际模型采集、学习器执行和评分均为零，保留未知原因、旧重放失败、负结果与封存。实际接入与采集仍需独立注册和授权。
