# Count-only observation-budget feasibility — 2026-10-17

🟢 [Done] The preregistered forecast finds that additional observations fit the acquisition/write ceiling **only under explicitly fixed rates and recurring known addresses**. A conservative allowance for newly appearing addresses exceeds the state ceiling before the planning targets are reached. This is a conditional budget calculation, not evidence of predictive improvement or permission to acquire data. Existing negative results and all seals remain unchanged.

## Frozen scope, assumptions and ceilings

Protocol `b57ad7e9bdbed3c9a59d18d6b76bdac89098a44694171ebcaa178cf75501b579` and code were registered before running the forecast. It pins 296 prior JSON/gzip evidence artifacts. Inputs are the same pooled and joint evidence checkpoints used in the [support-versus-effect audit](R2_CASAS_COUNT_SUPPORT_AUDIT.md): 17,553 fitted observations and 52,659 historical changed writes each. Their exact current-table hashes remain equal. No raw source, confirmation scores, development or test data are read; new learning and acquisition are both zero.

The exact current raw event, weekday, hour and 15-minute clock bin remain unchanged, as do the detail representations, prior learner results and strict evidence rule:

```
abs(branch_positive/n - complement_positive/m)
    > sqrt(log(80)/(2*n)) + sqrt(log(80)/(2*m))
```

The fixed planning ceilings are **263,295 additional qualified observed outcomes**, **789,885 additional changed writes per arm**, and **100,000 count routes**. These permit at most 15 extra copies of the original cohort, giving total scale k<=16. They are explicit resource allowances, not measured data availability. Outcomes are the qualified delayed observations from the original task, not arbitrary raw sensor rows; the count tables cannot supply raw-row yield, duration or bytes required. Three historical writes per outcome remain the accounting rule for any hypothetical future learning.

## Two separate counterfactuals

**Observed-rate uniform recurrence:** multiply every fitted contingency cell by the same positive integer k. This fixes both conditional outcome rates and current/history frequencies. The rate contrast remains constant and the original radius decreases with support. For every branch, integer inversion finds the smallest k that passes the strict original rule, then checks that k passes and k−1 fails using the original `evidence()` implementation. Missing complements and exactly zero observed contrast have no finite multiplier under this assumption. This does not establish that the corresponding population contrast is zero.

This is arithmetic for hypothetical new observations with exactly repeated rates. It does not duplicate training data into a model, count repeated records as new evidence, or perform learning. It assumes stationarity and recurrence that these saved counts cannot verify. The existing bound remains a proxy, not calibrated time-series significance.

**Fixed-margin per-branch oracle:** preserve k*n, k*m and the current's total positives, then hypothetically reassign positives to maximize one branch's contrast at a feasible endpoint. These are optimistic branchwise requirements. Different branch assignments need not coexist in one contingency table, so their summed eligible mass is only an upper bound on individual branch eligibility. The implementation deliberately reports no opposing-context count, global required scale or global feasibility for this oracle. It is not a replacement learner or an acquisition plan.

The observed-rate planning targets are fixed before execution: at least **ten exact current contexts with distinct passing branches in opposite directions**, at least **four directional outcomes** for each such branch, and at least **1% of fitted observation mass** in passing branches. Every additional cohort is uniform; no branch-targeted selection is allowed. These are planning targets, not substitutes for the original confirmation coverage, active-day, independent accuracy, Brier, timing-ablation or shuffled-feedback gates.

## State bounds

Under a **closed address world**, every original address simply recurs, so the route count stays fixed. Under an **open address world**, an extra outcome can introduce at most one current address and one detail address; global already exists. Thus a conservative upper bound after B extra outcomes is `S + 2*B`. This is a worst-case bound, not a prediction of actual growth. New addresses also violate the exact recurrence assumption, so the two worlds must not be combined into a guarantee of useful predictions.

The state-safe full-cohort multiplier is `1 + floor((100000−S)/(2*N))`, capped separately by the acquisition/write ceiling. Both models allow only k<=3 whole cohorts under this conservative state allowance.

## Retained result

| Observed-rate forecast | Four-branch pool | Original joint |
| --- | ---: | ---: |
| Existing count routes | 9,219 | 15,322 |
| Minimum total cohort scale for planning targets | **4** | **5** |
| Required additional qualified outcomes | **52,659** | **70,212** |
| Forecast additional writes per arm | 157,977 | 210,636 |
| Acquisition/write ceilings satisfied | Yes, conditionally | Yes, conditionally |
| Closed-world routes at required scale | 9,219 | 15,322 |
| Open-world route upper bound at required scale | **114,537** | **155,746** |
| Conservative state ceiling guaranteed | **No** | **No** |
| State-safe extra-outcome allowance | 45,390 | 42,339 |
| Largest state-safe whole-cohort scale | 3 | 3 |

The per-arm minima are planning summaries, not an equal-update comparison. A future comparison satisfying both projected targets would need a common scale of at least k=5: **70,212 extra outcomes and 210,636 extra writes per arm**, subject to a separate preregistration and the unresolved state conditions. In this audit both arms acquire zero outcomes and execute zero learning writes.

At k=3, pooled passing fit mass is only 139 observations (below 1% of 17,553) and seven opposing contexts (below ten). Joint has 60 observations and zero opposing contexts. At the full acquisition allowance k=16, closed-world pooled eligibility reaches 2,576 observations (14.68%) and 241 opposing contexts; joint reaches 3,496 (19.92%) and 360. These remain fitted-count projections, not confirmation coverage or successful predictions. The worst-case route bounds at that allowance are 535,809 and 541,912, exceeding capacity.

For observed rates, **3,969 pooled branches** have no finite scale (2,140 missing complements, 1,829 zero contrasts). Joint has **6,423** (1,359 missing complements, 5,064 zero contrasts). An oracle does not solve absent alternatives or all-one-class contexts: its nonfinite counts are 3,928 pooled and 6,388 joint. At k=16, the oracle's individually eligible observation-mass upper bounds are 45.69% and 50.23%; these cannot be interpreted as simultaneous achievable coverage.

**Decision:** the closed-world acquisition/write forecast passes its planning targets, while the conservative open-world state guarantee fails for both models. The result does not establish real-world infeasibility, because the route bound is conservative; it also does not authorize claiming feasibility without a bound on new addresses. No collection, new learner, factor growth or runtime integration is started. No prior negative result is reclassified as positive.

## Verification, artifacts and seals

128 related tests pass. Validation includes strict integer inversion versus direct evidence scans on small count tables; positive/zero-contrast and missing-complement cases; directional outcome support; distinct-context accounting; exact three-write forecasts; read-only model conservation; and the separation of independent oracle assignments from global feasibility. Fresh-process replay exactly matches the result and deterministic compressed branch trace. All 296 prior JSON/gzip artifacts remain hash-identical.

Protocol, registration, consumed execution marker, result and replay receipt live under `data/processed/benchmark_fixtures/casas_observation_budget_*_v1.json`. The branch trace is `casas_observation_budget_branches_v1.json.gz`. No model is created or modified. Replay needs only the tracked fit checkpoints:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/casas_observation_budget.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_casas_observation_budget.py
```

Exclusive artifact creation prevents silently replacing consumed runs. No raw rows or sealed predictions are opened; the prior 4,249,759-record seal and older household seals remain in force without rescanning the source.

🟠 [Next] Preregister a count-only finite-vocabulary state ceiling before any acquisition decision. Distinguish recurring exact addresses, new clock/history combinations within the observed raw-event vocabulary, and genuinely unseen raw events. Keep current/context conditions unchanged, make every vocabulary assumption explicit, and retain an unresolved open-world status for unbounded new events. Do not add an unknown-event rejection policy, change learning inputs or treat a state bound as predictive usefulness without a separate experiment.

🔴 [Later] New data collection and independent learner validation require a separately justified preregistration and the original positive-component/quality conditions. Physical memory, energy and calendar acquisition estimates require evidence unavailable in these counts.

日本語: 保存済みカウントだけで追加観測量と予算を事前登録して計算しました。観測率・既知アドレスの再来を仮定すると、必要な追加観測は集約52,659件、非集約70,212件で取得・更新上限内です。しかし新しい状態が増える保守的上限では必要時の状態数が114,537・155,746となり、100,000セル以内を保証できません。これは予測改善でも取得可能性の実証でもありません。実際の取得・学習は0、根拠閾値・現在状態・既存の負の結果と封印は維持しています。

简体中文：仅使用保存计数，预注册并计算额外观测及预算。在观测率不变、已知地址重复的假设下，聚合需额外52,659条、非聚合70,212条合格结果，均在采集和更新预算内。但新状态增长的保守上界分别达到114,537和155,746，无法保证低于100,000单元。这不是预测改进或实际可采集性的证明。实际采集与学习均为0，证据阈值、当前状态、既有负结果和封存全部保持。
