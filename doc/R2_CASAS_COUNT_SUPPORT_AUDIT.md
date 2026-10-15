# Training-count support-versus-effect audit — 2026-10-15

🟢 [Done] A preregistered, read-only decomposition separates missing alternatives, insufficient sample support, fixed outcome-margin constraints and insufficient observed rate separation. It uses only the saved 80-day HH104 fitting counts from the pooled and joint evidence arms. It changes neither the evidence threshold nor the exact current context. Earlier negative results remain negative; no new learner, source, confirmation prediction or sealed predictive field is opened.

## Registration and unchanged conditions

Protocol `d0587212aba030cd442145a604e6e488f104ef2c7ecd2bacadc31a9ff6c5183b` and code hashes were registered before computing this decomposition. Prior zero-activation results were known; this is a descriptive audit of already consumed training, not a fresh evaluation. The registration pins 290 earlier JSON/gzip evidence artifacts, including both input checkpoints and every previous negative result.

Inputs are exactly `models/casas_pooling/pool_evidence_v1.json.gz` and `joint_evidence_v1.json.gz`. Both contain the same 17,553 fitted observations and 52,659 historical changed local writes. Their current-table SHA is identical: `4e7a84d1aaa8b757ccf1dde3d7dc2e2366580b33d5f5badd2701e28e8346319d`. The stored raw sensor/message, weekday, hour and 15-minute bin are retained exactly. No current addresses are pooled, dropped or redefined.

Each detail partition must sum to its exact current counts; all currents must sum to the global counts; the checkpoint's actual changed writes must equal three times its fitted observations. The audit restores validated scalar counts without delivering any outcomes or invoking learning. New learning writes are **zero for both inputs**. No raw-source path or extraction is used. Byte hashes of older artifacts are checked before and after, without interpreting their confirmation scores.

## Five exhaustive categories

Let D be a branch's negative/positive counts, C=current−D its disjoint observed complement, n=sum(D), m=sum(C), and P the total current positives. The unchanged strict rule is:

```
observed separation = abs(D_positive/n - C_positive/m)
radius = sqrt(log(80)/(2*n)) + sqrt(log(80)/(2*m))
activate iff observed separation > radius
```

An observed branch must have n>0. Classification is ordered and exhaustive:

| Category | Fixed condition | Meaning |
| --- | --- | --- |
| `no_complement` | m=0 | No observed alternative history inside this exact current context. |
| `support_impossible` | n,m>0 and radius>=1 | Even the maximum possible rate difference of 1 cannot exceed the strict bound at these sample sizes. |
| `margin_impossible` | radius<1 but attainable maximum<=radius | Sample sizes permit a unit contrast, but the observed total class counts do not permit a sufficient contrast under any allocation. |
| `observed_below_bound` | Attainable maximum>radius, observed difference<=radius | A sufficient contrast is mathematically possible at these sizes and margins, but the fitted observations do not show it. |
| `evidence_pass` | Observed difference>radius | The original evidence rule passes. |

For the fixed-margin maximum, the detail-positive allocation x ranges over the integers from `max(0,P−m)` to `min(n,P)`. The maximum of `abs(x/n−(P−x)/m)` occurs at one of these endpoints. This avoids attributing rare-outcome constraints to weak observed association. Exhaustive small-count tests verify the endpoint formula against every feasible allocation and compare classification against the original `evidence()` function.

`observed_below_bound` is not proof of a small population effect, adequate statistical power or absence of temporal information. The bound is an unchanged proxy under serially dependent data, not calibrated significance. End-of-fit aggregate counts cannot identify temporal drift, online activation coverage or which histories will recur later. No such claims are made.

## Results

Each input has 3,667 exact current contexts. Counts below are **branches / fitted observations assigned to those branches**:

| Category | Original joint (11,654 branches) | Four-branch pool (5,551 branches) |
| --- | ---: | ---: |
| No observed complement | 1,359 / 1,432 | 2,140 / 4,370 |
| Support makes passage impossible | 10,087 / 14,248 | 3,200 / 9,947 |
| Fixed class margins make passage impossible | 139 / 1,013 | 99 / 1,245 |
| Attainable, but observed separation below bound | 69 / 860 | 112 / 1,991 |
| Evidence passes | 0 / 0 | 0 / 0 |
| Total | 11,654 / 17,553 | 5,551 / 17,553 |

Missing alternatives plus sample-size impossibility cover **98.22% of joint branches** and **96.20% of pooled branches**, with **89.33%** and **81.56%** of fit observations respectively. Pooling increases the observation share whose sizes and class margins permit passage from **4.90% to 11.34%**, but every such observed branch still falls below the unchanged bound. This is a descriptive shift in count coverage, not predictive improvement or a causal estimate of pooling's benefit.

Pooling also removes disjoint alternatives in some current contexts: the no-complement observation share increases from 8.16% to 24.90%. The remainder cannot all be labeled weak effects: fixed-margin impossibility accounts for 5.77% of joint observations and 7.09% of pooled observations. The 69 and 112 attainable-but-below-bound branches span 54 and 59 distinct current contexts respectively. Neither model has an evidence-passing or opposing-evidence context.

Category branch counts and assigned observation masses partition their totals exactly. Unique current-context counts by category are reported separately in the result; they overlap and must not be summed. Complements are reused across branches and are not independent samples. The audit counts each fitted observation once via its own detail branch, rather than summing overlapping complement sizes.

## Verification, artifacts and seals

The runner validates decompression size, checkpoint identity/schema, integer counts, address partitions, global/current/write conservation, cross-input current hashes and zero audit writes. A deterministic compressed trace records every branch classification and its unchanged current key, n, m, radius, observed contrast and attainable maximum. A separate JSON result contains the fixed summaries. **121 related tests pass.** Fresh-process replay exactly matches the entire summary and compressed branch trace; before/after source model hashes must remain unchanged.

Artifacts are under `data/processed/benchmark_fixtures/casas_count_audit_*_v1.json`, with branch trace `casas_count_audit_branches_v1.json.gz`. No model is generated or modified. The audit needs only tracked checkpoints, so replay works without raw CASAS data:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/casas_count_support_audit.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_casas_count_audit.py
```

Exclusive creation of protocol/registration/execution/results prevents silent reruns from replacing consumed evidence. Existing records remain sealed, including the [pooling study's 4,249,759 date-only excluded records](R2_CASAS_POOLING_FEASIBILITY.md). This audit opens **zero** raw rows and zero confirmation/development/test predictions; its seal status does not rely on another source scan.

🟠 [Next] Preregister a count-only observation-budget feasibility analysis at the same fixed current addresses and evidence rule before another learner or household comparison. Bound the additional support needed under explicit fixed-margin/rate assumptions, separate fixed-rate-assumption limitations from support deficits, and impose a hard acquisition/state budget. A forecast is not evidence that more data will improve predictions; do not select new features, a threshold or a winner from confirmation scores.

🔴 [Later] A new representation, data collection or independent validation requires a separately justified preregistration and the positive component prerequisite. No runtime/factor promotion follows from this audit.

日本語: 保存済み訓練カウントだけを事前登録した5区分で監査し、更新は両群とも0回でした。集約モデルの5,551枝は、比較相手なし2,140、支持不足3,200、クラス比率による制約99、成立し得るが観測差が不足112、根拠通過0に分かれます。支持・比較可能性の不足と観測差の不足を区別できましたが、母集団の効果が弱いとは断定しません。既存の負の結果・根拠閾値・現在状態・封印をすべて維持します。

简体中文：仅用已保存训练计数，按预注册的五类完成审计，两组新增学习更新均为0。聚合模型5,551分支分为：无可比较历史2,140、支持不足3,200、类别边际约束99、理论可达但观测差不足112、证据通过0。支持与可比较性不足已和观测差不足区分，但不能据此断定总体效应弱。保留既有负结果、证据阈值、当前状态条件与全部封存。
