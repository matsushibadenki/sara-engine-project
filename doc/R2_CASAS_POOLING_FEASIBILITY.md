# Bounded temporal-history pooling feasibility — 2026-10-15

🟢 [Done] A single four-branch representation was preregistered and evaluated entirely inside the already consumed HH104 fitting period. It reduces stored count routes but fails the registered real-data identification, coverage and usefulness prerequisites. This is a retained negative, not a new generalization result. The previous HH104 confirmation and every independent development/test period remain sealed.

## Frozen representation and experiment

Protocol `ce7a65b4ed1ee81dcc832a2d70f931fcd38f31777ab034072c58335518d3f9d0` and code hashes were saved before the new pooling execution. The registration pins 272 earlier JSON/gzip evidence artifacts. No new household was acquired. Prior HH104 scores were already known; this is an explicitly exploratory, reused-training component diagnostic.

The current address remains exactly the last raw sensor/message, weekday, hour and 15-minute clock bin. Within that address, a history maps to two causal booleans:

- Is the immediately previous sensor identifier the same as the current sensor identifier?
- Is the latest observed gap less than 10 seconds (existing bins 0/1)?

These yield at most four detail branches per current context. Older sensor identities, older intervals and exact recent gap bins are discarded by this fixed map. There is no outcome-dependent clustering, routing search or threshold sweep. The first boolean is a lagged sensor-repeat feature; the arm name `pool_order` does not imply that it retains arbitrary sequence order.

The evidence gate is unchanged from [HH104's earlier negative](R2_CASAS_EVIDENCE_DIAGNOSTIC.md): for branch counts D and disjoint complement C=current−D, require positive n=sum(D), m=sum(C) and rate separation greater than `sqrt(log(80)/(2*n)) + sqrt(log(80)/(2*m))`. Otherwise predict the exact global → current → current posterior, with prior strength 8. Serial dependence means this is an evidence proxy, not a calibrated significance guarantee.

All 13 arms were registered together: the five sufficient references (current, finer current, histogram, transition, joint), ungated pool, gated pool, unchanged joint evidence, lagged sensor-repeat-only pool, timing-only pool, constant-gap gated pool, shuffled-feedback gated pool and scalar duplicate. Constant gap fixes the fast boolean to false, matching the previous bin-2 ablation. Shuffled feedback rotates the fitting labels by half the sequence. The scalar duplicate uses the same scalar implementation and tests exact integrity, not an SNN advantage.

Every observed delayed outcome changes exactly three local integer counts, including the detail count when its gate is inactive. Prediction receipt capacity is 32, extraction pending capacity 16 and count-route capacity 100,000. The pooled candidate also has at most `1 + 5 × observed current contexts` routes. Reporting counters and offline identification summaries do not influence predictions and are not learned state. There are no matrices, GPU operations, backpropagation or model-wide scans in normal updates.

## Splits, prerequisites and seals

Only the previous 120-day fit interval is available. Its first 80 recorded dates fit the models (2011-06-15 through 2011-09-02); the remaining 40 dates are internal confirmation (2011-09-03 through 2011-10-12), with no learning. All predictive fields from 2011-10-13 onward are discarded by date. These dates were already used by earlier training; the new internal split is not untouched independent data. Source byte integrity still covers the entire compressed member without interpreting sealed predictive fields.

The original task and source conditions are unchanged: eight-event history, gaps <=300 seconds, cutoffs at least 60 seconds apart, first observed binary result 300–600 seconds later, Kitchen ON positive, split purging, no missing-result negatives, both classes on enough dates, repeated current states and at least ten original rich-current opposing-history groups. Failure of a source prerequisite stops before scores.

The new feasibility gates are fixed before execution:

- At least ten fitted current contexts must have distinct pooled branches passing the unchanged evidence rule in opposite directions relative to their shared current rate, with at least four outcomes of each branch's direction.
- Evidence must activate for at least 1% of internal-confirmation predictions on at least ten dates.
- Four-branch, route-capacity and actual-write limits must hold.

All registered training-only scores are reported even if these new feasibility gates fail; failure blocks independent validation. The original quality gates remain: daily Brier gain >=0.01 and balanced-accuracy gain >=0.02 over the best of the five sufficient references, constant-gap loss >=0.005, shuffled-feedback loss >=0.01 and exact scalar trace. Both feasibility and quality must pass; lower storage alone is insufficient.

## Retained measurements

The source prerequisites pass: fit has 17,553 examples (4,298 positive / 13,255 negative) with both classes on all 80 dates; confirmation has 9,801 examples (2,190 / 7,611), both classes on all 40 dates. All 19 observed raw current states recur; 702 rich contexts have support at least eight and 50 original opposing-history groups exceed the required ten. There are 5,574 late outcomes, excluded rather than made negative, and two unresolved outcomes.

All 13 arms perform **52,659 actual changed writes** (17,553 × three), zero confirmation updates and at most five pending receipts. The pooled model stores 9,219 routes, versus 15,322 for joint, a 39.83% reduction in this logical state count. It has 3,667 current contexts and 5,551 detail branches; maximum four branches per context, within the registered 18,336-route ceiling. This does not measure physical memory or energy savings.

| Arm | Mean daily Brier ↓ | Balanced accuracy ↑ |
| --- | ---: | ---: |
| Current | 0.155376022 | **60.0935%** |
| Finer current | 0.156827840 | 58.1287% |
| Histogram | 0.155468965 | 57.7377% |
| Transition | 0.155683354 | 57.3953% |
| Joint | 0.156492884 | 55.2709% |
| Ungated four-branch pool | 0.155891472 | 58.6612% |
| Gated four-branch pool | 0.155376022 | **60.0935%** |
| Joint evidence | 0.155376022 | **60.0935%** |
| Lagged sensor-repeat-only pool | **0.155153234** | 60.0423% |
| Timing-only pool | 0.156051232 | 58.4757% |
| Constant-gap gated pool | 0.155376022 | **60.0935%** |
| Shuffled-feedback gated pool | 0.191323959 | 50.4889% |
| Scalar duplicate | 0.155376022 | **60.0935%** |

**Feasibility fails:** fitted opposing-evidence contexts are **0/10**, activation **0/9,801** and active confirmation dates **0/10**. Bounds pass. The candidate exactly matches the current reference's entire prediction trace. Daily Brier gain, balanced gain and constant-gap loss are all zero; shuffled loss is 0.035947937 and scalar equivalence passes. Since the gate never activates, shuffled loss reflects disruption of the fallback current counts and supplies no evidence for a temporal mechanism. The small descriptive Brier reduction for the lagged sensor-repeat-only control is below the frozen 0.01 requirement and is not a selected replacement candidate.

Controlled synthetic witnesses do identify opposite futures from the same current context: fitted pooled predictions straddle 0.5, exact checkpoint replay holds, and the constant-gap control loses this timing distinction. Thus the implementation can distinguish the supplied witness, while the preregistered real-data prerequisite fails. This does not prove the source lacks temporal information, nor isolate whether effective support, conditional effect size or the fixed representation is the limiting factor.

## Reproduction and next decision

Protocol, registration, consumed execution marker, result and replay receipt are under `data/processed/benchmark_fixtures/casas_pooling_*_v1.json`. Final readouts are under `models/casas_pooling/`. Fresh-process refit and checkpoint prediction replay are exact; **113 related tests pass**. All 272 earlier JSON/gzip evidence artifacts remain hash-identical. The pinned raw HH104 member is required for replay; no new download is needed.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/eval/casas_pooling_diagnostic.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_casas_pooling.py
```

Exclusive artifact creation prevents silently overwriting consumed executions. Replay verifies code, protocol, all preserved evidence, exact refit results and saved readouts. **4,249,759 records** from the previous confirmation and later independent periods remain date-only sealed in this experiment, including all 3,845,589 independently sealed records from the previous run.

🟠 [Next] Stop this four-branch hypothesis. Before another model or household comparison, preregister a training-only support-versus-effect audit using the existing count states: separate branches that cannot meet the fixed evidence bound even at maximal rate separation from those with sufficient support but weak observed rate separation. Keep exact current context, the evidence threshold and the seals fixed; report a diagnostic rather than selecting a new candidate from confirmation scores.

🔴 [Later] Another representation, fresh household comparison, factor growth and runtime integration require a justified new preregistration and a positive component prerequisite. Do not promote storage reduction as predictive usefulness.

日本語: 既に使用したHH104の訓練120日だけを80日学習・40日内部確認に分け、最大4枝の集約を事前登録しました。13群の実更新は各52,659回で一致し、人工例の異なる未来は識別できました。一方、実データの反対方向の根拠を持つ文脈とゲート有効化はともに0で、実現性・有用性は不合格です。既存の負の結果と4,249,759件の封印を保持します。次は既存の訓練カウントだけで、支持不足と効果の弱さを切り分ける監査を事前登録します。

简体中文：仅使用已消耗的HH104训练120天，按80天拟合、40天内部确认预注册最多4分支的聚合。13组各执行52,659次实际更新，人工样例能区分不同未来。但真实数据中相反方向证据上下文及门控启用均为0，可行性与效用条件未通过。保留既有负结果和4,249,759条封存记录。下一步仅用现有训练计数预注册支持量与效应强度的区分审计。
