# Captured count-mixture and local lookup audit — 2026-10-27

🟢 [Done] A preregistered finite audit covers **all 24,000 route-key cells at 160 saved boundaries**, plus **1,500 key mixtures across ten captured exploration windows**. It separates old counts, recent counts and the original smoothing prior in conditional local lookup values. Exact identities, 170 tests and fresh-process pure replay pass. No learner, model lookup method, generator, seed decoder or prediction scorer runs. Causal explanations, old replay failure, scientific negatives and seals remain unchanged.

## Registered evidence and coverage

The [selection](../data/processed/benchmark_fixtures/route_count_mixture_selection_v1.json) was saved at **21:04:29 JST**, before implementation. [Code/tests registration](../data/processed/benchmark_fixtures/route_count_mixture_registration_v1.json) at **21:09:30 JST** preceded retained execution and a separate-process pure replay. This is retrospective descriptive algebra, not a new effectiveness experiment.

Only the existing v3 collection, its analysis/evaluator IDs and its final proof are parsed: **651,488 bytes**, under fixed 1 MiB individual/aggregate input limits. The existing collection/proof, actual capture order and read-only checks are validated without another capture. **471 prior file hashes** and registered source pins verify; other preserved files are read only for hashes.

Each snapshot has 45 routes and **150 canonical route-key cells**: 15 single-input routes × two keys, plus 30 two-input routes × four keys. The result keeps a lossless dictionary of **1,461 distinct `(n0,n1)` pairs**, all 160 × 150 cell references, exact lookup fractions, original floating values and rounding residuals. Different supports with the same probability are not merged. All 24,000 cells here have positive support, ranging 2–518 per key. The schema's zero-support case remains guarded: the captured vector does not encode dictionary-key allocation, so its actual lookup is unknown, not an invented 0.5 prediction.

The ten existing 180-receipt pre-start/pre-commit intervals yield 450 route-window summaries and 1,500 key mixtures. All 45 routes still show 36 new observations per interval. The eight focus keys belong to the same three already-recorded missing dependencies; they do not create a new selected task or altered gate.

## Original conditional lookup, counted once

For a captured key with positive support `N=n0+n1`, the unchanged local rule is:

```text
p(y=1 | key, captured table) = (n1+1)/(N+2)
                            = N/(N+2) × (n1/N)
                            + 2/(N+2) × 1/2
```

This is a **conditional lookup value if that key were queried in that route**. It is not a newly generated event prediction, proof the route was selected, or a claim about the true event probability. The audit computes the formula directly from saved counts; it does not call a model's `probability` method or emit predictions.

For old counts `a0,a1` and captured window increments `b0,b1`, let `A=a0+a1`, `B=b0+b1`, `N=A+B`. Then:

```text
old weight    = A/(N+2)        old contribution    = a1/(N+2)
recent weight = B/(N+2)        recent contribution = b1/(N+2)
prior weight  = 2/(N+2)        prior contribution  = 1/(N+2)

weights sum to 1
end lookup = old contribution + recent contribution + prior contribution
```

The original one pseudo-count per class is included **once**. Averaging two separately smoothed old/recent probabilities would generally duplicate or misweight the prior. A zero-weight group's empirical rate is unknown, not zero. Where recent support exists, the exact deviation from its empirical fraction is:

```text
p_end − recent_empirical
  = old_weight × (old_empirical − recent_empirical)
  + prior_weight × (1/2 − recent_empirical)
```

If old support is zero, the first contribution is zero with an unknown old empirical rate. Endpoint lookup change is reported only when both endpoints have positive support. An allocated-cell formula at zero total support may equal 1/2 mathematically; that is labeled conditional arithmetic and is never substituted for an unknown actual readout.

The [pure implementation](../src/sara_engine/evaluation/route_count_mixture.py) uses exact rational identities and explicitly retains the original binary64 division residual. No probability tolerance, changed pseudo-count, forgetting, reset, window-only smoothed rule or alternative learner is introduced.

## How mixing appears in the missing-dependency cells

The two seed 1237 cases have the same captured route 3 cells. Counts are ordered `[class0,class1]`; key indices refer to the saved canonical key order.

| Case / route / key | Old counts | Recent counts | End counts | Old conditional lookup | End conditional lookup |
| --- | --- | --- | --- | --- | --- |
| Both seed 1237 cases / 3 / 0 | [9,9] | [19,0] | [28,9] | 1/2 | **10/39 ≈ 0.25641** |
| Both seed 1237 cases / 3 / 1 | [16,2] | [0,17] | [16,19] | 3/20 | **20/37 ≈ 0.54054** |
| Interacting 1201 / 32 / 0 | [49,59] | [8,0] | [57,59] | 6/11 | **30/59 ≈ 0.50847** |
| Interacting 1201 / 32 / 1 | [55,56] | [0,10] | [55,66] | 57/113 | **67/123 ≈ 0.54472** |
| Interacting 1201 / 32 / 2 | [32,46] | [0,12] | [32,58] | 47/80 | **59/92 ≈ 0.64130** |
| Interacting 1201 / 32 / 3 | [24,55] | [6,0] | [30,55] | 56/81 | **56/87 ≈ 0.64368** |

All eight focus keys have unanimous recent labels and positive old counts of the opposite label. For example, route 32/key0 has eight recent class0 observations, yet the captured table's conditional class1 lookup is 30/59. Its deviation from recent empirical class1 rate 0 decomposes exactly into **59/118 old-positive contribution + 1/118 prior contribution**. This describes cumulative table contents; it is not a score of a newly queried prediction.

For route 3/key 1, the recent empirical class1 rate is 1. The end value 20/37 differs by **−17/37 = −16/37 old-count contribution −1/37 prior contribution**. Prior smoothing and old opposite-label mass are separated rather than conflated into generic “insufficient support.” No new per-key sufficiency threshold is imposed, and all original route-level support/quality gates remain unchanged.

## Count memory and error-weight memory are different

| Missing dependency | Old/recent/end observations | Old unweighted route-count share | Old share of time-weighted error normalizer |
| --- | --- | ---: | ---: |
| Independent1237 / route 3 | 36 /36 /72 | **50%** | **≈ 0.3891%** |
| Interacting 1201 / route 32 | 376 /36 /412 | **≈91.2621%** | **≈45.3404%** |
| Interacting1237 / route 3 | 36 /36 /72 | **50%** | **≈ 0.3891%** |

These quantities have different denominators and meanings. The rightmost column is the old share of `weight_sum`, not the fraction of actual accumulated error attributable to old observations. Key-specific smoothing weights differ again: route 3's old-key weights are 6/13 and 18/37, with prior weights 2/39 and 2/37. A small old share in the time-weighted error normalizer does not mean the probability count table has discarded old class counts.

The [previous score decomposition](SPARSE_ROUTE_SCORE_AUDIT.md) remains valid. This audit does not turn endpoint probabilities into the per-key losses that generated that historical score. Within-window query order, each pre-update probability and per-key prequential error allocation were not captured. Endpoints and a ledger digest cannot recover them. Thus the observed mixture is not a completed causal explanation of why a route lost ranking or why recovery failed. Reset/forgetting accuracy, score-rule interventions and functional substitution remain unknown.

## Verification and retained boundaries

**170 tests pass under CPython 3.10.22**: 22 new count/mixture tests and 148 retained pure audit/contract/comparison tests. They verify lossless 24,000-cell coverage, exact lookup/mixture/deviation identities, one-time prior counting, absent recent/old groups, zero-support uncertainty, invalid numeric/count types, independent count/weight shares, proof/hash/scope/negative mutations, no input mutation and consumed-attempt refusal. The process imports no learner. The static result reproduces exactly in a fresh process.

The [runner](../scripts/eval/route_count_mixture.py) freezes three inputs and all sources before execution and verifies them again afterward. Its result is **1,082,704 bytes**, under the preregistered 2 MiB result cap; six artifacts total **1,174,708 bytes**, below 4 MiB. The compact reference archive is lossless reporting, not a memory/CPU advantage of the learner. No new snapshots, model queries, outcomes, prediction scores, acquisitions or resource measurements are produced.

| Artifact | SHA-256 |
| --- | --- |
| Selection | `77b5c86f5e6c5fc18e87b6c10dce4d26238fb03f6df5a286b2cf867eccac47ce` |
| Code registration | `4fe6af117b9b993a4a9b16312c89cb478b72ae99f34d21816af6110003bd2f4a` |
| [Audit result](../data/processed/benchmark_fixtures/route_count_mixture_result_v1.json) | `58228b31e6595e02c897388831002ad49bf977e84a807600d0de1d625455fcc5` |
| [Pure replay](../data/processed/benchmark_fixtures/route_count_mixture_replay_v1.json) | `faa7e92b63b246adf87f723b7d84fd3f95667d92a825024fd49b17de3239fad8` |

```sh
# Captured-count algebra only; no learner/capture restart or new outcomes.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/route_count_mixture.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_route_count_mixture.py tests/test_route_score_audit.py tests/test_route_snapshot_contract.py tests/test_route_failure_audit.py tests/test_route_numeric_v2.py tests/test_reactivation_codec.py tests/test_project_paths.py
```

⭕️ [Pending] Causal separation of count-table history, probability adaptation and selection scoring would require a separately justified intervention protocol. None is authorized by this arithmetic audit. No automatic forgetting/reset, pseudo-count change, new trial or further audit loop is queued. Preserve the original negative recovery/retention results, old discarded replay unknown, current-state/evidence conditions and all seals. Independent-task, local-credit, resource/energy and production holds remain unchanged. Scalar conditional-probability algebra establishes no ANN-absent SNN benefit.

日本語: 採取済み160境界の24,000キーセルと1,500区間キー混合を、実装前に登録した規則で監査しました。過去・直近のカウントと平滑化を一度だけ加え、同じ確率でも支持数が異なる記録は統合せず保持しました。保持失敗3依存の過去カウント比率は50%／約91.26%／50%で、時間加重誤差の重みの過去比率とは異なりました。対象8キーは直近ラベルが片側に揃っても過去の反対ラベルが残り、局所確率に混在が表れていました。170テストと静的完全再現が成功し、471既存ハッシュを保全しました。学習器再実行・新規採点・条件調整はゼロで、因果原因と忘却・リセットの改善効果は未知のまま、旧失敗・負の結果・封印を維持します。

简体中文：按实现前注册的规则审计160个已采集边界的24,000个键单元及1,500个窗口键混合。过往、近期计数与原平滑先验仅计入一次，概率相同但支持数不同的记录不合并。3个缺失依赖的旧计数份额为50%／约91.26%／50%，不同于时间加权误差归一化权重的旧份额。8个焦点键的近期标签虽一致，旧相反标签仍在累计表中影响条件概率。170项测试及静态完全重放通过，保留471个既有文件哈希。不重跑学习器、不新增评分、不调条件；因果原因及遗忘/重置效果仍未知，保留旧失败、负结果和封存。
