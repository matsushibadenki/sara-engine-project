# Saved 160-boundary score decomposition audit — 2026-10-27

🟢 [Done] Preregistered and implemented a finite audit of the actual [v3 collection](SPARSE_ROUTE_CAPTURE_V3.md). All **480 boundary/dependency comparisons** and **30 exploration-window decompositions** pass exact arithmetic identities; a fresh-process pure replay matches. No learner, hook, evaluator seed decoder or prediction scorer runs. Scientific conditions, old replay failure, negative gates and sealed periods remain unchanged.

## Registration and scope

The [selection](../data/processed/benchmark_fixtures/route_score_audit_selection_v1.json) was saved at **20:49:03 JST**, before implementation; [source/tests registration](../data/processed/benchmark_fixtures/route_score_audit_registration_v1.json) at **20:53:33 JST** preceded retained execution and a separate-process pure replay. This is retrospective arithmetic on known evidence, not an independent effectiveness experiment.

The only parsed inputs are the saved 160-snapshot collection, its saved analysis/evaluator IDs and its numerical-noninterference proof: **651,488 bytes** combined. The fixed 1 MiB individual/aggregate input bounds are met. All **462 prior file hashes**, including original failures, source code, capture reports and markers, verify. Other pinned files are hash-read only.

Every captured boundary contributes its three current-phase evaluator dependencies, compared with the **third-ranked route of the same output** under the unchanged binary64 rule `weighted_error_sum / weight_sum + 0.005 × key_width`, with the original route-ID tie-break. This covers 160 × 3 = 480 pairs. These are repeated boundary observations, not 480 independent trials. All ten exploration windows contribute three additional endpoint decompositions. The three already-recorded missing dependencies are a fixed focus within that full population.

## Exact decomposition without double counting

Let `d` be the dependency, `b` its cumulative third-place comparator, `E`/`W` the stored weighted error/weight sums, and `λ` the **exact binary64 value of the unchanged coefficient 0.005**. The [pure audit](../src/sara_engine/evaluation/route_score_audit.py) preserves the original floating scores/ranks and separately uses exact rational representations of the stored numbers:

```text
cumulative error gap = E_d/W_d − E_b/W_b
width gap            = λ × (width_d − width_b)
actual float margin  = cumulative error gap + width gap + rounding residual
```

The residual accounts for original binary64 division, addition and subtraction. It is recorded explicitly, with **no equality tolerance, rounding-away or rational reranking**. Exact fraction strings remain in the result alongside actual floating scores. Residuals are nonzero in 475 of the 480 pairs; this is why mathematical component sums must not silently replace the original float comparison. Signed/typed original values remain in their unchanged captured evidence.

For a matching pre-start/pre-commit pair, use `E0,W0` before the original 180-receipt window and `E1,W1` afterward. Exact differences derive only from captured endpoints. The cumulative gap has this valid partition:

```text
old contribution    = E0_d/W1_d − E0_b/W1_b
window contribution = (E1_d−E0_d)/W1_d − (E1_b−E0_b)/W1_b
cumulative gap      = old contribution + window contribution
```

An alternative, equally exact description is:

```text
window-only error gap = (E1_d−E0_d)/(W1_d−W0_d)
                      − (E1_b−E0_b)/(W1_b−W0_b)
actual float margin   = window-only error gap
                      + (cumulative gap − window-only error gap)
                      + width gap + original rounding residual
```

**Cumulative and recent errors are not two independent addends.** Old-weight fractions differ by route, so a common mixing coefficient cannot be assumed. A negative normalized window contribution is not proof of a favorable window-only rate. The audit retains both identities and route-specific old-weight fractions.

The temporal comparisons keep the **same cumulative comparator** `b`. The existing window diagnostic's own third-place route/rank is listed separately. Its float subtraction/division has its own explicit rounding residual; it is not substituted for the original cumulative policy. No windowed/reset learner or width-cost variant is executed, and no outcomes are re-scored.

## Recorded exclusion margins

Numbers below are rounded **for display only**; stored identities use exact fractions.

| Missing dependency / cumulative cutoff | Cumulative error gap | Width gap | Original float margin |
| --- | ---: | ---: | ---: |
| Independent 1237: route 3 vs 27 | +0.0625263681 | **−0.005** | +0.0575263681 |
| Interacting 1201: route 32 vs 14 | +0.0141551648 | **+0.005** | +0.0191551648 |
| Interacting 1237: route 3 vs 27 | +0.0625263681 | **−0.005** | +0.0575263681 |

For both route 3 failures, the dependency's one-channel width is already favored over the two-channel cutoff route. Its error gap outweighs that advantage. The two-channel route 32 is disadvantaged against a one-channel cutoff, but its cumulative error gap is also positive. These are fixed-pair component statements; they do not evaluate a policy with width cost removed or prove that changing that cost would repair retention.

| Focus pair | Old contribution | Window contribution | Window-only error gap, same pair | Cumulative minus window-only gap |
| --- | ---: | ---: | ---: | ---: |
| Both route 3 pairs | −0.0000960755 | +0.0626224436 | +0.0629099193 | −0.0003835513 |
| Route32 vs 14 | +0.1078499838 | −0.0936948191 | +0.0098860562 | +0.0042691086 |

The route 3 pairs have old-weight fractions of about **0.3891% / 0.3617%** for dependency/cutoff; the current weighted-error margin is dominated by window contribution. This does not remove past counts from the probability tables that produced those residuals.

For route 32/cutoff14, the fractions are about **45.3404% / 0.3709%**. Different normalization explains why its window contribution to the cumulative gap is negative while its window-only error gap is positive. The latter plus unchanged width cost remains a positive same-pair diagnostic margin of about **0.0148860562**. The window's own third-place comparator is **route 11**, with margin about **0.0136096295** and dependency rank 6; confusing that with the cumulative comparator 14 would mix distinct comparisons. Both route 3 cases remain window rank 4 with the same comparator 27.

This strengthens the earlier observation that neither a total-support deficit at the existing 16-observation gate nor ID tie-breaking explains the immediate exclusion. It does **not** isolate why probability/error history is unfavorable, prove that forgetting fixes it, or establish functional equivalence of alternate retained routes. Counterfactual accuracy and causal attribution remain unknown. All three exact-retention failures and all ten recovery failures stay negative.

## Verification, preservation and limits

**148 tests pass under CPython 3.10.22**: 21 new exact-arithmetic/saved-proof tests plus 127 retained pure contract/audit/codec/path tests. They verify all rational identities, same-comparator binding, original float ranks against saved diagnostics, coefficient fidelity, one-ULP visibility, self-pair zeros, ID tie-breaking, malformed snapshot/truth rejection, source/proof/hash/seal mutations, no input mutation and exclusive consumed attempts. The test process imports no learner.

The [runner](../scripts/eval/route_score_audit.py) enforces the finite three-input roster, byte limits and pre/post source hashes. Its result is **451,589 bytes**, below the registered 1 MiB result ceiling; six audit artifacts total **542,190 bytes**, below 2 MiB. Exact static replay passes. No new model state, observation, label, prediction, acquisition or resource measurement is produced.

| Artifact | SHA-256 |
| --- | --- |
| Selection | `95e5facb0a7f81e470d954cebc254dcae4163e013430ee77cc6c961c3608af29` |
| Code registration | `2cfac8737b7412997be682b10e1128c08258efda8a8f4ba337969e253c4bf006` |
| [Audit result](../data/processed/benchmark_fixtures/route_score_audit_result_v1.json) | `4a266e227afd16dea51d4400c7c57498f75e153dc33ae11776ac1d815c8db9b9` |
| [Pure replay](../data/processed/benchmark_fixtures/route_score_audit_replay_v1.json) | `3be201d194f8a8b63ef83bb27afe7d4343ad8dfbc4138e25d5572606aa8af0b3` |

```sh
# Saved-evidence arithmetic only; no study or collection restart.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/route_score_audit.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_route_score_audit.py tests/test_route_snapshot_contract.py tests/test_route_failure_audit.py tests/test_route_numeric_v2.py tests/test_reactivation_codec.py tests/test_project_paths.py
```

⭕️ [Pending] Any causal intervention or learning/score-rule change needs a separately justified protocol; this audit authorizes none. No automatic forgetting/width retuning, new trial or extra audit loop is queued. The old discarded replay remains unknown and failed, while v2/v3 numerical successes remain valid in their separate scopes. Independent task/causal-credit/resource/energy and production holds remain unchanged. Scalar score decomposition supplies no ANN-absent SNN advantage.

日本語: 採取済み160境界だけを使う有限監査を実装前に登録し、480比較・30区間の分解を実施しました。累積と直近誤差の二重計上を避け、比較相手を固定し、丸め差まで含め有理数で厳密照合しました。保持失敗3件中2件では幅コストが依存側を0.005優遇しても、誤差差約0.06253が上回りました。残る1件は誤差差約0.01416に幅0.005が加わりました。148テスト・静的完全再現が成功し、462既存ハッシュを保全しました。学習器再実行・条件調整・新たな予測採点はゼロで、因果原因・反実仮想の改善効果は未知のまま、旧失敗・負の結果・封印を維持します。

简体中文：在实现前注册仅用已采集160个边界的有限审计，完成480组比较与30个窗口分解。避免累计与近期误差重复相加，固定比较对象，并以有理数保留浮点舍入残差。3项保留失败中2项宽度成本已优惠依赖0.005，但约0.06253的误差差超过该优势；另1项约0.01416误差差加上0.005宽度项。148项测试及静态完全重放通过，保留462个既有文件哈希。不重跑学习器、不调条件、不新增预测评分；因果原因和反事实改善仍未知，保留旧失败、负结果和封存。
