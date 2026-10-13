# Training-only support-gated readout — 2026-10-13

## Decision

🟢 [Done] Preregister and execute a single fixed **support-8** temporal readout diagnostic inside previously consumed HH103 training data. Source conditions pass with 38 supported history contrasts and both classes on all internal confirmation days. Ten readouts, including two separately registered sufficient-reference supplements, execute **27,972 actual changed learning writes each**, zero confirmation learning writes and exact saved-state/fresh-process replay. The fixed rule fails Brier, balanced-accuracy and gap-necessity gates; correct feedback pairing and scalar equality pass. Keep this negative and every original development/test seal.

🟠 [Next] Preregister a **current-relative predictive-evidence gate using existing local counts**, on fresh training identities, before an independent validation run. Raw support count alone does not establish that a temporal branch improves its shared current-state predictor. Keep three changed writes, all sufficient references, due-ordered feedback and original source/quality thresholds. Do not sweep support thresholds or tune on consumed HH103 confirmation/development; a training-only pass is required before external promotion work.

## Scope and frozen internal split

This is a component diagnostic motivated by the already-known HH103 learner failure and training-state sparsity audit. All original 180 training days were previously consumed; **internal confirmation is not unseen independent evaluation**. Registration precedes new candidate scoring. The original HH103 development period and 1,831-day test tail are excluded before time/sensor/message parsing.

Use the original metadata's first 120 recorded dates for fitting and its remaining 60 original training dates for internal confirmation:

| Period | Boundary | Usable cases and class support |
| --- | --- | --- |
| Fit | 2011-06-15 <= time < 2011-10-14 | 9,324: 1,690 positive / 7,634 negative; 119 usable days/class |
| Internal confirmation | 2011-10-14 <= time < 2011-12-13 | 9,352: 886 positive / 8,466 negative; 60 days/class |
| Original development + test seal | 2011-12-13 onward | 5,008,622 records checked only by date |

Keep the existing future Kitchen-prefix ON notification task, exact current sensor/message + weekday/quarter-hour inputs, eight-event history, 300–600 second observed follow-up, one cutoff per 60 seconds, split/gap purging and all class/day/contrast thresholds. Qualification is **38/10 contrasts**, 16 repeated current states, 100% recurrence-row coverage and 314 rich-current contexts with at least eight observations. A recorded fitting date has no eligible decisions; usable class-day counts still exceed the original requirements. No source threshold is reduced.

## Fixed rule, local updates and registration limits

Fix detail support **8**, equal to the existing prior strength; there is no threshold grid or post-score selection. A branch with fewer than eight delivered fitting outcomes outputs the **exact current-control probability**, including its duplicate current posterior step. A branch with support >=8 outputs the ordinary joint posterior. Only past delivered local counts determine eligibility. No confirmation labels are used to activate branches.

Every fitting feedback changes three integer counts: global, shared current, detail. Even an inactive detail receives its normal local feedback; no attempted/no-op update accounting substitutes for changed counts. The receipt keeps its original three addresses when the output probability is replaced, and identity/time validation precedes outcome access. Sparse capacity admission probes only those three addresses. All controls use the same examples and **9,324 × 3 = 27,972 changed writes**; total across ten arms is 279,720. The 100,000-route, 32-prediction-receipt and 16-extraction-receipt ceilings remain.

The first registration specified eight arms: current, transition, joint, support-gated joint, constant-gap joint, support-gated constant-gap, identical scalar gated joint and shuffled-feedback gated joint. It inadvertently omitted finer-current and histogram references from the existing sufficient-reference set. **Keep that original registration and negative result unchanged.** A separate registration supplements both references before their scores are inspected, compares against the complete sufficient set and preserves every candidate prediction and threshold. This is an explicitly post-candidate reference supplement, not a claim that all ten arms were registered before candidate scoring. It does not weaken conditions or convert the first failure into success.

## Results with the complete references

| Arm | Daily mean Brier | Balanced accuracy | Changed writes |
| --- | ---: | ---: | ---: |
| Current | 0.079161 | **61.6121%** | 27,972 |
| Finer current (supplement) | 0.079772 | 59.1980% | 27,972 |
| Histogram (supplement) | 0.079242 | 60.2531% | 27,972 |
| Transition | **0.079053** | 60.9933% | 27,972 |
| Joint | 0.079411 | 59.9729% | 27,972 |
| Support-8 joint | 0.079055 | 60.8004% | 27,972 |
| Constant gap | 0.079053 | 60.9933% | 27,972 |
| Support-8 constant gap | 0.079249 | 61.5557% | 27,972 |
| Scalar support-8 | 0.079055 | 60.8004% | 27,972 |
| Shuffled-feedback support-8 | 0.110373 | 49.4297% | 27,972 |

Support gating improves daily Brier by 0.000356 over ungated joint, but is **0.00000150 worse than transition**; the required gain is 0.01. Its balanced accuracy is **0.8117 percentage points below current**, versus the required 2-point gain. Constant-gap gated Brier loss is 0.000194, below 0.005. Shuffled-feedback loss is 0.031318, above 0.01. Scalar predictions match exactly. Three quality/mechanism gates fail; two integrity/mechanism gates pass. No independent data is opened to rescue the candidate.

Only **769/9,352 (8.22%)** confirmation decisions activate a temporal branch; 8,583 use exact current fallback. Gated constant-gap activates 2,068 branches, since collapsing gaps changes local support counts. This destruction control measures the complete registered representation-and-gate behavior; its different coverage prevents treating the small difference as isolated evidence of timing necessity.

Final count states and arm/support policy metadata are stored under `models/casas_confidence/` as deterministic compressed JSON. Confirmation does not change any learned count/checkpoint; receipt, clock and diagnostic counters remain normal bookkeeping. Peak pending predictions is five; core route counts are 4,685–7,430. No matrix operations, GPU or whole-network gradients. No physical-energy, latency, factor-discovery or spike-specific advantage claim follows.

## Verification and reproducibility

96 related tests pass. They verify exact current fallback, support activation using only past feedback, replaced receipt identity and early-outcome rejection, distinguishable learned histories, original-period date-first sealing, all earlier artifact/code/model hashes, real-write counts, scalar equality and the complete-reference gate calculation. Independent CLI replays refit and restore identical count states and reproduce metrics/traces; no original development/test inputs are interpreted.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/casas_confidence_diagnostic.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/casas_confidence_references.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_casas_confidence.py tests/test_casas_180.py tests/test_sparse_count_replay.py tests/test_delayed_sparse_counts.py tests/test_casas_longitudinal.py tests/test_casas_notification.py tests/test_room_temporal.py tests/test_traffic_temporal.py tests/test_appliances_learning.py tests/test_appliances_prospective.py
```

Both execution identities are consumed; do not re-register or re-execute them. Protocol, registrations, markers and result receipts are under managed processed locations. Reuse the exact source-member acquisition receipt from [the qualified long-source experiment](R2_CASAS_180_DAY_COMPARISON.md). Old negatives and their frozen conditions remain unchanged. Support-8 is not adopted; no confidence-threshold retries on these identities.

日本語: 訓練期間だけで支持数8の読み出しを事前登録し、元の現在状態入力・実更新回数・遅延条件を維持して比較しました。10群各27,972回の実更新と再現性は成立しましたが、強い現在状態対照を上回らず、性能・時間情報除去の条件は未達です。元の開発・テスト期間を開かず、固定支持数だけの再調整を停止します。

简体中文：仅在已使用的训练期预注册支持数8读取，保持当前输入、真实更新预算和延迟条件。十组各27,972次真实局部写入与精确重放成立，但未超过强当前状态对照，性能与时间消融门槛未通过。原开发期和测试期保持封存，停止在这些身份上调整支持数阈值。
