# Hand-authored numeric delta preflight — 2026-10-29

🟢 [Done] Separately preregistered **nonlearning numeric before/after verification** checks strict types, finite ranges, exact increments, no-op rejection and unchanged fields. **72 literal-fixture transactions**, **2,592 active-field checks**, **28,728 unchanged-field checks**, **40/40 rejected faults**, **233 pure tests** and exact fresh-process replay pass. **515 historical artifact hashes** verify. No actual learning, sampled task inputs/outcomes, scoring, model capture, runtime scalar comparison or sealed access occurs.

## Registration and independence of fixture values

The immutable [selection](../data/processed/benchmark_fixtures/route_numeric_delta_preflight_selection_v1.json) was written before implementation and includes all twelve hand-authored numeric rows. Code/tests and pure dependencies are frozen before the new execute/replay phases. The [parent symbolic delta-ledger](SPARSE_ROUTE_DELTA_LEDGER_PREFLIGHT.md), scientific conditions, update-address contract, original gates, negatives and seals remain unchanged. No old phase is resumed or modified.

Before and after numbers come from literal tables, not a learner, generated stream, numerical update function or prediction/error scorer. A fixture constructor places the literals into the fixed 435-slot catalog. The validator independently computes the expected delta relation to inspect those supplied views; it never applies a numeric update. Fixture rows cover ticks **0,64,1023**, four count/error factor combinations, two engine descriptors and three roles: **12 ×2 ×3 =72** standalone transactions.

The shared before table has mass2 in each class/key slot, error8.0, weight16.0 and route observations8/16 for width1/2. Active after observations are literal9/17. Remaining slots retain exact predecessor values. All numeric rows below were registered before code:

| Tick | Count history | Error weighting | Selected mass after | Error after | Weight after | Independent error-increment witness |
|---|---|---|---:|---:|---:|---:|
| 0 | cumulative | uniform | 3 | 8.25 | 17.0 | 0.25 |
| 0 | cumulative | public_exponential | 3 | 8.25 | 17.0 | 0.25 |
| 0 | discounted | uniform | 3 | 8.25 | 17.0 | 0.25 |
| 0 | discounted | public_exponential | 3 | 8.25 | 17.0 | 0.25 |
| 64 | cumulative | uniform | 3 | 8.25 | 17.0 | 0.25 |
| 64 | cumulative | public_exponential | 3 | 8.5 | 18.0 | 0.5 |
| 64 | discounted | uniform | 4 | 8.25 | 17.0 | 0.25 |
| 64 | discounted | public_exponential | 4 | 8.5 | 18.0 | 0.5 |
| 1023 | cumulative | uniform | 3 | 8.25 | 17.0 | 0.25 |
| 1023 | cumulative | public_exponential | 3 | 8200.0 | 32784.0 | 8192.0 |
| 1023 | discounted | uniform | 32770 | 8.25 | 17.0 | 0.25 |
| 1023 | discounted | public_exponential | 32770 | 8200.0 | 32784.0 | 8192.0 |

These are fresh, hand-authored algebraic examples. They are not claimed to be feasible complete trained prefixes, and they do not continue any old run. Key/mass-side placeholders retain the parent's schema-fixture meaning. No actual input bits or result labels are read or generated.

## Types, finite values and exact equality

The [pure verifier](../src/sara_engine/evaluation/route_numeric_delta_preflight.py) requires:

- Masses and route observations are **exact Python integers**; bool and float coercion are refused.
- Error and weight sums are **exact finite Python floats**; integer/string substitutions, NaN and both infinities are refused.
- All fields are nonnegative. Negative zero is explicitly refused; it cannot hide an unchanged-field bit-pattern difference.
- Route observations are at most205; masses/errors/weights are at most6,717,440, the unchanged parent finite fixed-allocation bound. Per-route error sum cannot exceed its weight sum.
- The full immutable canonical catalog has435 slots, including both class sides, every key and every route scalar. No missing/extra/duplicate/mutable address view is accepted.

Equality uses typed fingerprints and **exact `float.hex()`**, with no tolerance, rounding of fixture values, field dropping or type normalization. This retains distinctions such as int8 versus float8.0, even though ordinary Python numeric equality would equate them. Signed zero is rejected before comparison.

## Increment and unchanged-field checks

For the nine pinned active routes, check observed mass increments by one or public `2**floor(t/64)` according to the count rule; route observations increment by one; the weight normalizer increments by1.0 or the public float weight according to the error rule. Every required after value must be strictly larger than its before value.

The squared-error numerator is checked against a separately supplied **positive finite float witness**, bounded by the route's error weight. Its after value must equal the exact rounded binary64 sum `before + witness`, and must also be greater than before. Thus a tiny positive addition absorbed by rounding is rejected as a numeric no-op. The test suite explicitly exercises the smallest positive subnormal added to8.0, which rounds back to8.0.

The error witness is **trusted fixture input**, supplied independently of the after view. It is not recomputed from a model prediction or authenticated outcome in this phase. A mutually consistent but fictitious after/witness pair is not evidence of correct actual learning. Binding that witness to real probability/result provenance remains untested.

Typed fingerprints bridge the numeric views into the unchanged symbolic structural adapter. It verifies receipt/source/key/side identity, one-tick delivery, same owner, diagnostic equality and the exact **36 changed /399 unchanged** address set. This bridge validates a temporary metadata ledger only; no numeric value is committed to a learner. The numeric verifier is read-only and checks an independently supplied trusted predecessor before inspecting differences.

Independent whole-view conservation also checks nine observation increments, nine count-weight increments and nine normalizer-weight increments. No opposite-class, inactive-key or inactive-route write can be hidden by matching a total. A 1 ULP change in an active error or inactive float is rejected; there is no tolerance to absorb it.

## Faults and input preservation

All **40 registered fault cases** are rejected, and original immutable fixture views and witnesses remain unchanged. Coverage includes:

- Bool/float substitutions for integer fields; integer/string substitutions for float fields.
- NaN in before or after, positive/negative infinity, negative values and negative zero.
- Wrong mass/observation/error/weight increments and individual no-op fields.
- Opposite-class, inactive-route and inactive-key changes; pinned credit-key drift.
- Wrong shuffled source, same-tick result and diagnostic drift during core verification.
- Absorbed positive float increment, nonfinite/nonpositive/overweight error witness and stale trusted predecessor.
- Missing/duplicate/mutable numeric field views and foreign ownership.

Twelve new pure tests also exercise valid numeric bounds, explicit 1 ULP perturbations, frozen input immutability, unchanged hand-authored rows and refusal of changed scope. With 221 earlier pure checks, **233 tests pass**. No learner/generator/scorer module is imported or run. The two engine names are fixture descriptors; their actual implementations are not executed or numerically compared.

## Evidence, limits and preserved outcomes

The [runner](../scripts/eval/route_numeric_delta_preflight.py) verifies **515 historical artifact hashes**, parent binding, all source pins and Python 3.10.22. The result is **4,849 bytes**, below the fixed 256 KiB result cap; all new artifacts remain below 2 MiB. Separate fresh processes produce exact canonical replay. Consumed phases refuse restart before recomputation; verification recomputes supplied-fixture arithmetic only.

All current actual-learning, outcome-read, generated-sample, prediction-score, new-snapshot and resource-measurement counters are zero. The 2,592 active-field checks are inspections of hand-authored after values, not actual writes performed by an engine. Same-role runtime scalar equality, observed local updates and adaptive-system acceptance remain untested.

No threshold, seed, source, delay, phase boundary or scientific condition is adjusted. The old tuple/list reproduction failure, unknown discarded-run equality, 256-tick recovery failures, independent-task/resource negatives and sealed periods remain intact. This scalar read-only arithmetic uses no GPU, matrix or global-backprop dependency, but establishes no SNN-exclusive mechanism, energy benefit, production readiness or component promotion.

- [selection](../data/processed/benchmark_fixtures/route_numeric_delta_preflight_selection_v1.json): `f12f582674601d3574563951d0d6493f91fd2eb82db84f02421d58f8c5cb30a2`
- [registration](../data/processed/benchmark_fixtures/route_numeric_delta_preflight_registration_v1.json): `5d439cc87703564eab310ff305c54962926f42cd57557d88afa818635d0b7e88`
- [result](../data/processed/benchmark_fixtures/route_numeric_delta_preflight_result_v1.json): `61595227cc5cf210a82dca10052c9af58a4467d7d4c12e62b6984014f5c4ffc5`
- [replay](../data/processed/benchmark_fixtures/route_numeric_delta_preflight_replay_v1.json): `99126d1dd6520cc7bd72ef58861caea3dc8ebce07f379eed02b610cbfc7af0c9`

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/route_numeric_delta_preflight.py verify
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m unittest discover -s tests -p test_route_numeric_delta_preflight.py
```

🟠 [Next] Preregister a finite, read-only probability/error-witness binding contract using hand-authored records only: connect prior-state local probabilities, key/class credit and public error weights to expected numeric increments. Keep fixture witness consistency distinct from authentic runtime outcome provenance; no learner, task stream, scoring or sealed access.

⭕️ [Pending] Actual numeric delta observation, authenticated event/key/result provenance, independent scalar implementation/equality, fresh independent-task/SNN causal evidence and resource promotion remain on hold for a separately authorized version. No automatic learning or old restart.

## 日本語

手作業で固定した12数値テンプレートを72条件組合せに配置し、型・有限性・正しい増分・無変更を検査する非学習の数値検証器を事前登録・実装した。boolの整数扱い、NaN・無限大・負値・負のゼロ、誤増分、無変更、非対象欄の1 ULP変更など40故障をすべて拒否した。233テスト・別プロセス再現一致・515件の過去ハッシュを確認。2,592件は手書き状態の検査であり、実学習の更新回数ではない。誤差増分は独立の検証用入力との照合に限定し、実予測・結果との結び付けやスカラー実装の数値一致は未検証とした。旧失敗・負の結果・封印と元の条件を維持する。

## 简体中文

预注册并实现了不执行学习的数值差分验证器，将12个人工定义的数值模板用于72组条件，检查类型、有限性、精确增量与未改变字段。40种故障全部拒绝，包括布尔/类型替换、NaN、无穷、负值、负零、错误增量、无变化及非目标字段1 ULP变化。233项纯测试、独立进程精确重放和515个历史哈希验证通过。2,592次字段检查不是实际学习更新。误差增量只与独立提供的样例见证值核对，未验证真实预测/结果来源或标量实现数值等价。保留旧失败、负面结果、原始条件和封存。
