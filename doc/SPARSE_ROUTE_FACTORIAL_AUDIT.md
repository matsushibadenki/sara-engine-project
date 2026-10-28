# Saved factorial rank and allocation audit — 2026-10-28

🟢 [Done] Preregistered, pure saved-report audit of **20 cases × four cells × four boundaries**, including all **48,000 key cells**, **240 allocation intervals** and **480 paired boundary comparisons**. No learner, generator, model lookup or new prediction scorer is imported or executed. Exact fresh-process replay and **178 pure tests** pass. Existing failures, original thresholds and sealed periods remain unchanged.

## Registration, finite scope and evidence

The [selection](../data/processed/benchmark_fixtures/route_factorial_audit_selection_v1.json) was saved before implementation. Its scope sentence mistakenly said 60 pair comparisons per boundary; the immutable [scope correction](../data/processed/benchmark_fixtures/route_factorial_audit_scope_v1.json), also saved before implementation, fixes the arithmetic to **20 cases × six pairs = 120 per boundary / 480 total**. All cases and pairs were already mandatory in the selection; budgets and conditions did not change. Neither file is overwritten.

Registration freezes the new pure audit code/tests, correction, Python 3.10.22 and project-path utility before execution. Only the v4 primary, byte-identical replication and proof are parsed. Historical files are otherwise read for hash preservation only. **484 prior hashes** verify, including the old reproduction failure, earlier negatives and every v4 artifact. Inputs stay within 8 MiB per file / 16 MiB aggregate; the result is **2,850,239 bytes**, below the 4 MiB result cap, and the complete audit artifact set stays below 8 MiB.

The audit independently validates the fixed case/cell/boundary catalog, all key probabilities from raw masses and public-time priors, effective masses, original score ranks, observation conservation, interval updates, final dependency retention and two-block recovery. All four saved boundaries are after delayed outcome and before next narrowing. Boundary 180 correctly has 45 pool routes; later snapshots have nine. Interval allocation totals are exactly `9 × elapsed_receipts`; this is not new learning.

For each output, the audit identifies the lowest `(saved_score, route_id)` route inside the saved pool with at least **16 route observations**, following the fixed original readout rule. These are reconstructed boundary choices, not predictions on new inputs. A key may still abstain if its saved dictionary cell is absent; absent probabilities remain `null` and are never filled with 0.5. The 16-observation rule is route-level support, not a claim of 16 observations for every key.

## Probability changes and adaptive allocation

Holding public-time error weighting fixed, cumulative versus discounted evidence differs in **local probabilities and ranks at the first saved boundary (180)** in all ten changed cases. At this boundary the warmup schedule gives equal route-observation counts and the pool is still the full catalog. Thus probability/rank differences exist before any observed allocation divergence, but these coarse boundaries do not identify the exact first divergent tick.

By boundary **512, before any changed label has been processed**, nine of the ten changed-case pairs already have different route allocations and pools. The tenth first differs at saved boundary 768. This is critical: v4's final gains include changes in the earlier adaptive selection trajectory. They cannot be attributed solely to forgetting old opposite-class counts after the change. The experiment establishes a paired total algorithm effect on known streams; this audit does not isolate a direct effect under fixed per-route support.

The stationary counterparts show the same nine allocation divergences by 512 under exponential errors; the remaining stationary pair never differs in allocation at a saved boundary. All retain accurate predictions. With uniform errors, evidence-history pairs differ in probabilities/ranks at 180 in all ten cases, but only eight pairs have different allocation by 512. These are descriptive saved-boundary comparisons, not additional trials.

All six pairwise comparisons and per-key probability differences, including equal-raw-mass cases, are retained. Equal raw weighted masses do not imply equal underlying observation histories across differently weighted cells. The audit does not treat such matches as causal controls.

## Original support and complete dependencies

Across the ten changed cases there are 30 true dependencies. Every cell has **30/30 above the unchanged 16-observation route threshold** at boundaries 512, 768 and 1024. The following separates pool membership from the route actually eligible to win the output readout:

| Cell | At 768: dependencies in pool / selected | At 1024: dependencies in pool / selected |
|---|---:|---:|
| Cumulative / exponential error | 27 / 23 | 27 / 27 |
| Discounted / exponential error | 30 / 30 | 30 / 30 |
| Cumulative / uniform error | 5 / 0 | 5 / 4 |
| Discounted / uniform error | 5 / 0 | 7 / 3 |

Counts are out of 30, not complete-case successes. Both uniform-error cells still retain all required dependencies in **0/10 cases**. A correctly ranked individual route or an improved local probability is insufficient for complete dependency retention. Four cumulative/exponential dependencies are in the pool at 768 but do not win the output readout; pool retention alone is also insufficient. No missing-route failure here is reclassified as insufficient original route-level support.

The interval 768→1024 directs exactly **one third of route updates to the three true dependencies in all ten discounted/exponential cases**: 768 of 2,304 route updates per case. In the cumulative/exponential cell seven cases reach this share; the three old retention failures allocate two ninths, 512 of 2,304. These adaptive allocation differences are part of the total intervention effect, not matched per-route training evidence.

## Recovery remains negative

Discounted/exponential exploration starts at ticks **519–525** and commits at **699–705**, preserving the 180-receipt exploration length. Cumulative/exponential starts at 518–525 and commits at 698–705; evidence discounting does not consistently trigger earlier. These are saved lifecycle metadata, not inferred from boundary ranks.

In every discounted/exponential changed case, the block ending at **704** remains below 0.90 exact accuracy (range **0.078125–0.5**). The block ending at **768** passes (0.984375–1.0), and the block ending at **832** passes at 1.0. Therefore the earliest two consecutive qualifying blocks end at **832**, giving **320 ticks after change**, beyond the unchanged **256-tick** gate. A good boundary readout at 768 is not equivalent to two consecutive accurate preceding blocks.

This observation does not prove the old recovery gate impossible, nor authorize shortening exploration, moving the change, loosening the gate, shifting block alignment or rerunning an old phase. Uniform-error recovery stays `null`; the original cumulative/exponential seed-1201 interacting failure remains `null`. Old reproduction failure, all original scientific negatives and real-task/resource holds remain intact.

## Implementation, validation and limits

The [pure module](../src/sara_engine/evaluation/route_factorial_audit.py) has no dependency on learner/evaluator modules. Eight new tests cover input immutability, full coverage, no learner imports, identical-pair identity, the pre-narrow warmup boundary, corrupt ranks/masses/probabilities/observations/case identities, and rejection of imputed recovery. These pass together with the 170 existing pure audit/codec/path tests: **178 tests**, without executing the v4 development tests or any learner. New execute/replay phases are consumed, exclusive and refuse restart; read-only verification recomputes saved arithmetic only. No model capture, scoring, data acquisition, resource measurement or tuning occurs.

This scalar count-table mechanism audit is compatible with SARA's local, sparse learning constraints. It supplies no evidence of an advantage exclusive to spiking networks over an event-driven scalar nonspiking implementation, and no energy, production or independent-task claim. Adaptive allocations prevent direct causal attribution to probability changes alone.

## Retained evidence

- [selection](../data/processed/benchmark_fixtures/route_factorial_audit_selection_v1.json): `fe67f5e06ed23237ce5cbb8db4dde0dd2087d959094a5d4486a2898079089952`
- [scope](../data/processed/benchmark_fixtures/route_factorial_audit_scope_v1.json): `9f38d7038b018ca7f9badf2cc46ce713a62378936e74f4f427d45dd275a917bd`
- [registration](../data/processed/benchmark_fixtures/route_factorial_audit_registration_v1.json): `3f3d98bbfcf74bdda4faf944b20fb466233ac88a2391564211f776545afd50ae`
- [result](../data/processed/benchmark_fixtures/route_factorial_audit_result_v1.json): `cf61ab26204fbda4547c471721b92336db3f5c54a9ae88acaa497fad09bc2f13`
- [replay](../data/processed/benchmark_fixtures/route_factorial_audit_replay_v1.json): `605c3fd2c02efd3410d83a569c2ee382d3fe31b2dfee6cacbacafef9f376a2fe`

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/route_factorial_audit.py verify
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest --basetemp=workspace/route_factorial_audit/pytest tests/test_route_count_mixture.py tests/test_route_score_audit.py tests/test_route_snapshot_contract.py tests/test_route_failure_audit.py tests/test_route_numeric_v2.py tests/test_reactivation_codec.py tests/test_project_paths.py tests/test_route_factorial_audit.py
```

🟠 [Next] Design and preregister a **nonexecuting fixed-allocation intervention contract** that separates local evidence effects from selection feedback: identical externally fixed active-route receipts, independent count/error factors, actual local-write equality, delayed outcomes, original recovery/retention gates and a strong event-driven scalar control. State explicitly which questions a fixed schedule can answer and which adaptive recovery claims it cannot. Do not run learners or change scientific conditions during this design step.

⭕️ [Pending] A fresh independent task and justified SNN-specific causal mechanism remain prerequisites for further effectiveness/resource promotion. No automatic extra experiment, acquisition, threshold tuning or sealed access.

## 日本語

保存済み報告のみで320境界・48,000キーセル・240観測配分区間・480条件間比較を監査した。誤差時間加重を固定すると、履歴減衰の有無で全10件が180受信時点の確率・順位から異なり、9件は変化前の512受信時点ですでに更新配分も異なる。したがって改善を局所確率だけの直接効果とは断定できない。改善条件は768時点で真依存30/30を選べるが、704終了ブロックは未達で、連続2ブロックが揃うのは832時点、回復320 tickとなる。256 tickの負の結果、旧再現失敗、封印を維持する。178テスト・純粋再計算の完全一致を確認。次は固定配分での介入契約を設計・事前登録するだけとし、追加学習や条件調整は行わない。

## 简体中文

仅使用已保存报告审计320个边界、48,000个键单元、240个更新分配区间和480组条件比较。固定误差时间加权时，10组在180次接收时均有概率和排名差异，其中9组在变化前512次接收时已有更新分配差异，不能把最终改善单独归因于局部概率。改进条件在768边界选择全部30个真实依赖，但704结束的区块未达标，连续两个合格区块到832才齐全，因此恢复仍为320 tick，未通过256 tick门槛。178项测试与纯重放通过，保留旧失败、负面结果和封存。下一步只设计并预注册固定分配的干预契约，不执行学习器或调整条件。
