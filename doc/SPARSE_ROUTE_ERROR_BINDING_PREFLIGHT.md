# Read-only local probability/error witness binding — 2026-10-29

🟢 [Done] Separately preregistered **hand-record arithmetic binding** connects prior-state local probabilities, allocation/key/class credit and public error weights to expected increments. **11 local records**, **72 numeric fixtures /648 route bindings**, **20/20 rejected faults**, **242 pure tests**, exact fresh-process replay and **521 preserved historical hashes** pass. No learner, task stream, performance scoring, actual update, runtime scalar comparison or sealed access occurs.

## Scope and registration

The immutable [selection](../data/processed/benchmark_fixtures/route_error_binding_preflight_selection_v1.json), including all eleven literal local records and ordered fault cases, was saved before implementation. New source/test registration precedes new execute/replay phases. The [numeric-delta preflight](SPARSE_ROUTE_NUMERIC_DELTA_PREFLIGHT.md), old reports, source files and consumed attempts remain untouched. All scientific conditions and original adaptive gates remain unchanged.

The earlier checker trusted a supplied error-increment witness. This stage verifies that witness against the hand-authored prior state, credited class and original public-weight equation. It is still fixture consistency, not authenticated actual learning evidence. Arithmetic on a hand-entered probability/result side is not an evaluation of a model prediction or a new benchmark score.

## Exact lookup/error equations

For an allocated key, read both masses from the **trusted before view**:

```text
prior = 1                           for cumulative counts
prior = 2**floor(receipt_tick/64)    for discounted counts
p1 = (n1 + prior)/(n0+n1+2*prior)
y = 0 for mass_left, 1 for mass_right
weight = 1.0 or float(2**floor(receipt_tick/64))
error_increment = weight*(p1-y)**2
```

The operation order matches the registered original scalar residual rule. Probabilities and increments compare by exact `float.hex()` with strict finite float types and no tolerance. The count masses remain nonnegative bounded integers; bool substitution is rejected.

For an **unallocated key**, require zero masses and `pre_probability=None`. Only its update-residual calculation uses fallback0.5; the lookup remains abstention. An allocated zero-count key instead has actual smoothed probability0.5. Allocation is therefore supplied as an explicit, separately matched fixture witness; zero counts alone do not reveal dictionary allocation. No probability is fabricated for a missing key.

## Hand-authored records

| Tick | Count rule | Error rule | Allocated | n0 | n1 | Credit side | Recorded probability | Recorded increment |
|---|---|---|---|---:|---:|---|---:|---:|
| 0 | cumulative | uniform | True | 2 | 2 | mass_left | 0.5 | 0.25 |
| 0 | cumulative | uniform | True | 0 | 2 | mass_left | 0.75 | 0.5625 |
| 0 | cumulative | uniform | True | 0 | 2 | mass_right | 0.75 | 0.0625 |
| 0 | cumulative | uniform | True | 2 | 0 | mass_left | 0.25 | 0.0625 |
| 0 | cumulative | uniform | True | 2 | 0 | mass_right | 0.25 | 0.5625 |
| 64 | discounted | public_exponential | True | 0 | 4 | mass_left | 0.75 | 1.125 |
| 64 | discounted | public_exponential | True | 0 | 4 | mass_right | 0.75 | 0.125 |
| 1023 | discounted | public_exponential | True | 0 | 65536 | mass_left | 0.75 | 18432.0 |
| 1023 | discounted | public_exponential | True | 0 | 65536 | mass_right | 0.75 | 2048.0 |
| 64 | discounted | uniform | False | 0 | 0 | mass_left | None | 0.25 |
| 64 | discounted | uniform | True | 0 | 0 | mass_left | 0.5 | 0.25 |

These eleven examples cover asymmetric counts, both class sides, public priors/weights at64 and1023, and absent versus allocated-zero keys. They are isolated algebraic records, not trained prefixes or sampled inputs/outcomes.

## Integrated credit and numeric-delta verification

The twelve parent literal numeric templates are placed in two engine descriptors and three roles, producing **72 standalone fixtures**. Both masses in each prior key are2, so their recorded probability is0.5. Error witnesses retain the hand-authored quarter-residual values under each public weight. After values are literal templates, not learner-produced updates.

New fixture credits use the common hand-entered output-side vector `(mass_left, mass_right, mass_left)`. Every one of the three active routes for an output must credit that same side. Before calling the unchanged numeric verifier, the new [binding module](../src/sara_engine/evaluation/route_error_binding_preflight.py) verifies all nine record positions, keys, class sides, allocation flags, before-state probabilities and independently supplied error increments. The parent then verifies the full 435-slot type/range catalog, exact 36 active changes, 399 unchanged fields, provenance, one-tick delay, diagnostic isolation and conservation.

The older numeric-only fixtures deliberately alternated mass-side placeholders to test field updates. They remain valid examples for that narrower checker and are preserved verbatim. They do **not** establish a common three-output result delivery; an explicit regression test confirms they still pass the older numeric verifier but are rejected by this new common-output binding. This extends interface coverage without relabeling any old experiment, altering original thresholds or treating schema placeholders as observed outcomes.

The integrated fixtures cover allocated keys. Absence and allocated-zero behavior are covered in the separate local records; no full runtime dictionary-allocation lifecycle is tested or inferred.

## Fault rejection

All **20 preregistered faults** are rejected with immutable inputs unchanged:

- Wrong prior probability, probability from an after state, wrong prior clock, wrong class side, wrong error weight or wrong increment.
- Bool/NaN probabilities, missing seen-key probability, invented absent-key probability and absent keys with positive mass.
- Wrong key/order, allocation flag drift and numeric witness drift.
- Inconsistent class credit within one output, wrong shuffled source, same-tick delivery, stale prior state, reordered records and wrong common output vector.

The predecessor probability is recomputed from the trusted before masses, rather than accepted because the after increment happens to agree with a supplied value. Neither the binding function nor the parent numeric checker mutates a learner or fixture.

## Verification and limits

Nine new tests combine with 233 prior pure tests: **242 tests pass**. No learner, generator or prediction scorer module is imported or executed. The two engine names remain fixture descriptors; neither numerical engine runs. Same-role runtime scalar equality is still untested.

The result is **4,105 bytes**, under the fixed 256 KiB per-artifact limit; all six new artifacts remain below 2 MiB. The [runner](../scripts/eval/route_error_binding_preflight.py) verifies 521 prior hashes, parent binding, all sources and Python 3.10.22. Separate fresh processes reproduce the exact canonical result. Consumed phases refuse restart before computation; read-only verification recomputes hand-record arithmetic only.

Explicit allocation flags, before-state selection and class credit are **trusted hand-fixture inputs**, not authenticated runtime provenance. The records do not establish that a real pre-outcome lookup was captured before its result became available. In particular, prior symbolic adapters pin a complete fixture credit including side; this is schema validation, not an online API that may know the future class at receipt time. A real two-phase interface must seal receipt/key/allocation/probability first and add class credit only on its own delayed delivery. That causal boundary remains untested here.

All actual-learning, sampled-task, outcome-read, performance-score, new-snapshot and resource counters remain zero. No sealed period is opened. The old tuple/list reproduction failure, unknown discarded-run equality, 256-tick recovery negatives and independent-task/resource holds remain unchanged. Fixed-readout or shadow fixture checks never establish original adaptive acceptance. No SNN-exclusive benefit, runtime efficiency, energy or production claim follows from these scalar identities.

- [selection](../data/processed/benchmark_fixtures/route_error_binding_preflight_selection_v1.json): `2f2b41aceff9d00f82e56b5ffd109431c4d2b7654cf9e32ffcdf5b33de2bb543`
- [registration](../data/processed/benchmark_fixtures/route_error_binding_preflight_registration_v1.json): `a146de8aa813a5f44394712b1a9f84a5e4d265170070cfab08287bac96cc4f9d`
- [result](../data/processed/benchmark_fixtures/route_error_binding_preflight_result_v1.json): `fec2797cf2fbe2b4861c68a013d0aa4e427686cdc2827743534c1e60d430d5e1`
- [replay](../data/processed/benchmark_fixtures/route_error_binding_preflight_replay_v1.json): `7335ebaa3b9d4e9b7dace0c13ffb816437272bb3ab9368c617ac47690c2800ba`

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/route_error_binding_preflight.py verify
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m unittest discover -s tests -p test_route_error_binding_preflight.py
```

🟠 [Next] Preregister a finite nonlearning two-phase witness-seal contract using hand-authored metadata: seal receipt/key/allocation/pre-probability before any class credit, then bind class only to its own one-tick delayed result. Reject premature class knowledge, post-outcome replacement and receipt/source substitution. No learner, sampled outcomes or sealed access.

⭕️ [Pending] Authentic runtime allocation/event/result provenance, observed probability/error deltas, independent scalar numerical equality, independent-task/SNN causal evidence and resource promotion remain on hold for separately authorized work. No old restart or automatic training.

## 日本語

更新前の局所確率・割当情報・キーとクラス信用・公開時刻重みから誤差増分を照合する非学習の契約を事前登録・実装した。11局所記録と72件・648経路の数値検証、20故障の拒否、242テスト、別プロセス再現一致、521件の過去ハッシュを確認した。同一出力の3経路のクラス信用を一致させ、未割当のNoneと割当済みゼロカウントの0.5を区別した。以前の欄別検証例は保存し、共通の実結果を表す証跡には読み替えない。割当・クラス情報は手書きの信頼入力であり、実行時の由来や前後関係、数値エンジンの一致は未検証。追加学習・条件調整なしで、旧失敗・負の結果・封印を維持する。

## 简体中文

预注册并实现了只读契约，将更新前局部概率、分配信息、键/类别信用和公开时间权重绑定到误差增量。验证11个局部记录、72组/648条路径、20种故障拒绝、242项纯测试及独立进程精确重放，521个历史哈希保持一致。同一输出的3条路径必须获得相同类别信用，区分未分配的None与已分配零计数的0.5。旧字段样例保持原状，不重新解释为真实共同结果。分配与类别仍是人工信任输入，未证明运行时来源、时间先后或数值引擎等价。没有新增学习或条件调整，保留旧失败、负面结果和封存。
