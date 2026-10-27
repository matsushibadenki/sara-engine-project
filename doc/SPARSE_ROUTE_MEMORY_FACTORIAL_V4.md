# Independent local evidence/error weighting comparison v4 — 2026-10-27

🟢 [Done] A separately preregistered **2×2 experiment** separates count-history retention from error time weighting. On the fixed, already known synthetic cases, discounting evidence while retaining public-time-weighted errors improves final dependency retention from **7/10 to 10/10**, and changed-case mean tail/frozen exact accuracy from **94.53125% / 95.4296875% to 100% / 100%**. **All ten cases still fail the unchanged 256-tick recovery gate:** recovery takes 320 ticks. No promotion is authorized.

## Registration and execution

The [selection](../data/processed/benchmark_fixtures/route_memory_factorial_v4_selection_v1.json) was written before the new implementation. Development used only seeds 81001/81002. [Registration](../data/processed/benchmark_fixtures/route_memory_factorial_v4_registration_v1.json) froze code, tests, Python 3.10.22 and historical hashes before the primary phase. Each separately consumed primary/replication phase runs **80 models**: two tasks × five original study seeds × stationary/changed × four factor cells. Inputs and labels are identical across cells; each label arrives one tick after its receipt. These known seeds make this a retrospective paired intervention experiment, not independent confirmation.

No old runner is invoked, resumed, or modified. The unchanged-cell training metrics, original update/prediction ledger and frozen metrics match the saved original `error_triggered` arm exactly in all 20 cases. The registered tuple/list serialization binding for initial pools is the only layout normalization. New model-identity fields and state hashes are separately reported, not passed off as old full-state equality.

## Independent factors and matched actual updates

Let `w(t)=2**floor(t/64)`. Cumulative evidence increments the observed class mass by one. Discounted evidence increments that mass by `w(t)` and evaluates a seen key at public query tick `q` as:

```text
p(class 1) = (C1 + w(q)) / (C0 + C1 + 2*w(q))
effective evidence from receipt s = w(s)/w(q)
```

Thus the smoothing prior remains one effective observation per class. No table reset, class deletion, FIFO eviction, latent change hint or extra per-key timestamp is introduced. This is exponential discounting in 64-tick public steps, not finite-window truncation. Query time is the next receipt tick during training; frozen evaluation fixes it at 1023. Unseen keys abstain. Inactive stored masses are retained but discounted in lookup. Absolute masses remain bounded within this fixed 1024-receipt protocol; no unbounded-runtime claim follows.

Independently, error accumulation uses either `E += (p-y)^2; W += 1` or `E += w(t)*(p-y)^2; W += w(t)`. Every cell keeps `cost=E/W + 0.005*key_width`, the original 16-observation readout threshold, trigger, warmup, cooldown, exploration length, candidate catalog and selector. The shared 64-tick schedule is fixed without tuning.

Every receipt changes **nine distinct routes, three per output**. Every route update genuinely changes four learning fields: observed-class mass, route observation count, weighted squared-error sum and weight sum. Each cell/case has **9,216 actual route updates and 36,864 non-no-op learning-field changes**. Independent per-receipt expected tables, errors, weights and observation deltas check all active and inactive fields; mass conservation is verified after every receipt. Weighted count mass is explicitly distinguished from the number of observations. No dummy reset/update equalization is used. Detector/scheduler metadata writes and arithmetic operation counts are not matched; no runtime, memory or energy comparison is claimed.

Adaptive per-route allocations can differ after the intervention changes predictions and the trigger. Equal total update counts do not imply equal observations per route. The measured causal contrast is the total effect of the registered algorithm intervention on these deterministic streams, including that feedback; it does not uniquely identify a microscopic cause of the old failures.

## Results under unchanged thresholds

| Count history / error weighting | Changed tail exact mean | Changed frozen exact mean | Final dependency retention | Recovery ≤256 ticks |
|---|---:|---:|---:|---:|
| cumulative / uniform | 19.76562% | 26.83594% | 0/10 | 0/10 |
| cumulative / public_exponential | 94.53125% | 95.42969% | 7/10 | 0/10 |
| discounted / uniform | 37.03125% | 40.97656% | 0/10 | 0/10 |
| discounted / public_exponential | 100.00000% | 100.00000% | 10/10 | 0/10 |

All four cells have stationary tail exact accuracy 100% and zero false exploration starts. Uniform-error cells never satisfy the two-consecutive-64-block recovery definition by the end, recorded as `null`, not zero. With cumulative evidence/public-time error, nine cases recover in 320 ticks and interacting seed 1201 never recovers. With discounted evidence/public-time error, all ten recover in 320 ticks, so the original failure count remains ten.

Paired changed-case mean tail effects of evidence discounting are **+17.265625 percentage points** with uniform errors and **+5.46875 points** with exponential errors; the interaction on this metric is **−11.796875 points**. Public-time error weighting improves mean tail exact accuracy by **+74.765625 points** with cumulative counts and **+62.96875 points** with discounted counts. These are deterministic paired differences, without significance or independent-generalization claims. All individual cases, Brier contrasts and retention contrasts are retained in each report.

Historical permanent/shuffled controls are preserved as historical references; they are not re-executed as new matched factorial controls. This minimal four-cell mechanism comparison cannot assert passage of the original full permanent/shuffled component gate, even where final accuracy improves. Thresholds and original negatives remain unchanged.

## Local probabilities, ranking and retention

Read-only snapshots cover **180, 512, 768 and 1024 receipts**, after the delayed outcome and before the next narrowing step: **320 snapshots / phase**, all 45 routes and all 150 binary route-key cells each. Diagnostics include raw/effective masses, local probabilities, all-route scores, per-output rank, observations, pool membership and evaluator-only true-dependency IDs. Full learner-state hashes agree before/after every snapshot. These fixed boundaries are not the previous 160-boundary hook roster and do not claim to capture every new commit.

At the unchanged recovery deadline (boundary 768), the following are cumulative/exponential → discounted/exponential comparisons. Probabilities are shown in canonical binary-key order; rounded values below do not replace exact retained binary64 evidence.

| Previously missing dependency | Rank | In pool | Route observations | Local probabilities |
|---|---|---|---|---|
| Independent 1237, route 3 | 4 → 1 | no → yes | 72 → 141 | [0.256410, 0.540541] → [0.039020, 0.939331] |
| Interacting 1237, route 3 | 4 → 1 | no → yes | 72 → 141 | [0.256410, 0.540541] → [0.039020, 0.939331] |
| Interacting 1201, route 32 | 5 → 1 | no → yes | 412 → 139 | [0.508475, 0.544715, 0.641304, 0.643678] → [0.087738, 0.916582, 0.906416, 0.083984] |

These are deadline ranks; the old route-32 commit rank was eight and is preserved. By boundary 1024 all three are still rank one in the discounted/exponential cell. Uniform errors illustrate why local improvement alone is insufficient: route 3 reaches rank two and pool membership at the end under discounted/uniform, and route 32 reaches rank one under both uniform cells, yet **no uniform-error case retains all three required dependencies**. The complete reports preserve all four cells and all routes rather than only these historical failure examples.

## Verification, bounds and retained failures

Two fresh processes produce byte-identical canonical reports; proof verification is read-only and never invokes a learner. Each phase contains 81,920 training receipts. Each report is **6,536,561 bytes**, below the preregistered 8 MiB per-artifact cap; the full artifact set remains below 32 MiB. No resource measurement was performed. The new 12 tests cover factor independence, exact baseline behavior, delayed-label rejection, smoothing/time boundaries, independent actual-update ledgers, no-capture/capture equality on development seeds, read-only full snapshots, frozen scalar/packet equality, retained budgets and refusal of consumed primary/replication before model invocation. The 170 previous pure audit/codec/path tests pass separately: **182 total**. The pure-suite first invocation used the system temporary directory, causing two existing path-policy tests to fail before learner execution; rerunning with the required managed `--basetemp` passes all 170 without source changes.

**477 prior file hashes** verify, including the old failed replay and negative reports. The new discounted scalar oracle evaluates weighted masses with the fixed prior; the old cumulative-only oracle and old modules remain untouched.

The old tuple/list reproduction failure remains a failure; discarded old numeric equality stays unknown. Existing nonstationary, independent real-task, overlap and acquisition negatives remain intact. No real-data reads, acquisition, sealed test access, threshold changes, SNN-exclusive benefit, GPU/matrix/backprop dependency, resource promotion or deployment follows. This is local, sparse scalar evidence/error processing, compatible with the SARA constraints but not proof of a spiking advantage.

## Evidence and read-only reproduction

- [selection](../data/processed/benchmark_fixtures/route_memory_factorial_v4_selection_v1.json): `8b196ab28022353fc60a7ee4e5bc304c07b226b305fcf0c8249667a933413a1e`
- [registration](../data/processed/benchmark_fixtures/route_memory_factorial_v4_registration_v1.json): `bbbab2d8c6417977f582542fbd6dc0a46740e9fe64ee7c253bce1e1f7a0ddac9`
- [primary](../data/processed/benchmark_fixtures/route_memory_factorial_v4_primary_v1.json): `9468faeefd63665b4dcadb75ebc6ab6a84244139d8056021dbd2d8c213edd96d`
- [replication](../data/processed/benchmark_fixtures/route_memory_factorial_v4_replication_v1.json): `9468faeefd63665b4dcadb75ebc6ab6a84244139d8056021dbd2d8c213edd96d`
- [result](../data/processed/benchmark_fixtures/route_memory_factorial_v4_result_v1.json): `7c2f47834311f6faa0620480e287c71ce0435d575c5b511e6229b49e3c7af142`

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/route_memory_factorial_v4.py verify
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m unittest discover -s tests -p 'test_route_memory_factorial_v4*.py'
```

Primary and replication are consumed and refuse restart. Do not run the old experiment to reproduce this proof.

🟠 [Next] Before proposing another learner version, preregister a finite saved-report audit of the four cells' dependency/readout ranks and observation allocation at the fixed boundaries, distinguishing improved local probabilities from adaptive route-allocation feedback and the still-failed recovery deadline. Do not infer unobserved commit timing from these snapshots.

⭕️ [Pending] Independent-task usefulness, full component acceptance, SNN-specific causal advantage and resource/energy measurements remain on hold under their original requirements. No automatic extra run or schedule adjustment is authorized by this result.

## 日本語

履歴保持と誤差の時間加重を独立に切り替える4条件を事前登録し、各条件9,216回の経路更新・36,864フィールドの実変更を検証した。履歴減衰＋誤差時間加重では、変化後10件すべてで最終依存保持と末尾・凍結精度100%に改善した。一方、回復は320 tickで、元の256 tick基準は全10件で未達のままである。誤差時間加重なしは両条件とも完全依存保持0/10。旧実行の再開・条件調整・封印解除はなく、旧再現失敗と負の結果を保持する。既知seedでの機構比較であり、独立課題・SNN固有優位性・資源改善の証明ではない。次は保存済み報告だけで順位と観測配分を監査し、追加学習は行わない。

## 简体中文

预注册并验证了历史计数保留与误差时间加权的四种组合，每种条件均执行9,216次实际路径更新和36,864次实际学习字段变化。历史折扣加误差时间加权使10个变化案例的最终依赖保留及尾段、冻结准确率均达到100%，但恢复仍需320 tick，全部未通过原定256 tick门槛。无误差时间加权的两组均为0/10完整依赖保留。保留旧复现失败、负面结果和封存期，不重启旧运行或调整条件。该结果不证明独立任务有效性、SNN独有优势或资源收益。下一步仅审计保存的报告，区分概率、排名和观测分配反馈。
