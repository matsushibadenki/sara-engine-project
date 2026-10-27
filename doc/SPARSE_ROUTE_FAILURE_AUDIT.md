# Saved-report recovery and dependency-retention audit — 2026-10-26

🟢 [Done] A preregistered finite audit separates **observed timing constraints**, **reported accuracy**, **exact dependency membership** and **unrecorded causal explanations**. All 100 saved pool lifecycles reconcile. The original ten recovery failures and three exact-retention failures remain negative. No learner, generator, data acquisition, scoring or resource measurement is run.

## Fixed inputs and registration

The [selection](../data/processed/benchmark_fixtures/route_failure_audit_selection_v1.json) was written at **14:54:38 JST**, before implementation. [Code/tests registration](../data/processed/benchmark_fixtures/route_failure_audit_registration_v1.json) at **15:00:54 JST** preceded the retained execution and a separate-process pure replay. Outcomes were already known; this is a retrospective audit, not another confirmatory study.

The seven parsed inputs are the original selection, primary report and failed-replay receipt, followed by the numerical-v2 selection, primary report, replication report and final proof. They total **1,593,723 bytes**, within the fixed 2 MiB individual/aggregate parsing ceilings. Other preserved files are read only for hashes. The population is fixed at 20 cases × five arms; timing/retention details cover exactly ten changed error-triggered cases, with ten stationary controls and the saved periodic comparison.

The [pure evaluator](../src/sara_engine/evaluation/route_failure_audit.py) checks the numerical-v2 proof/report bindings and exact codec equivalence against the original saved reference with only its already registered provenance mapping. It preserves the original scientific conditions and failed-replay unknown. It imports no learner, generator or random/permutation reconstruction. The [runner](../scripts/eval/route_failure_audit.py) freezes all sources, bounds input reads, uses exclusive phase files, and supports exact pure replay. **432 prior file hashes** and registered source pins verify. The result is **95,443 bytes**, below its 256 KiB limit; six new artifacts total **160,425 bytes**, below 1 MiB.

## What timing explains, and what it does not

The audit rederives only the original two-consecutive-64-tick-block criterion at ≥90% from saved block summaries and confirms the saved recovery times and gate flags. No prediction is regenerated or re-scored. The change remains tick 512 and the recovery deadline remains tick 768.

| Recorded/derived timing across ten changed candidates | Result |
| --- | --- |
| First trigger after change | 6–13 ticks |
| Observed full-bank exploration before commit | 180 receipts |
| First new-pool commit after change | 186–193 ticks; absolute ticks 698–705 |
| Remaining time until original deadline | 63–70 ticks |
| Earliest end of two aligned blocks wholly after commit | 320 ticks after change in nine cases; 384 in one |
| Saved recovery result | 320 ticks in nine cases; absent in one |

There is insufficient remaining time for **two wholly post-commit blocks** in every candidate case. This identifies a timing/aggregation constraint without proving that the original gate was impossible: pre-commit or mixed blocks could also achieve ≥90%. Moreover, the old pool's tables and readout scores continue changing during exploration. “Before commit” does not mean a frozen or unlearning predictor. Shorter exploration, faster detection or a different rule could also change route quality; their causal effect is unknown here.

For the last deadline-eligible pair, `[640,704)` and `[704,768)`, the first block contains **58–64 pre-commit predictions** and only **0–6 post-commit predictions**. Its saved total correct count is only 4–11 of 64. The second block passes ≥90% in nine cases, but the first does not. This locates the recorded failure under the unchanged rule; it is not permission to move the deadline or redefine recovery.

Sharp subset-count bounds avoid inventing within-block predictions. For a stored correct count `C`, pre-commit count `O` and post-commit count `N`, post-commit correct predictions lie in `[max(0,C−O), min(C,N)]`. A zero-sized subset has unknown accuracy, not zero accuracy. In the interacting seed-1249 boundary, the second block contains one pre-commit and 63 post-commit predictions, with 63 total correct. Its post-commit correct count is therefore **62–63**, not an asserted exact score. Its saved qualifying pair ends at delay 320 even though the earliest pair wholly after its tick-705 commit ends at 384. These are compatible observations.

The interacting seed-1201 failure is different: the entirely post-commit `[704,768)` block has only **33/64 correct**, and its saved tail/frozen accuracies remain 45.31%/54.30%. Low observed quality persists after commit; it cannot be described merely as a lack of two post-commit measurement blocks. The report still lacks the route-level state needed to explain why those predictions fail.

The periodic control's first relevant window starts at tick 436, before the change, and commits at 616. Its 180 receipts include 76 before and 104 after the change. Two wholly post-commit blocks can end at the original deadline. All five independent periodic cases recover at delay 256, while none of the five interacting periodic cases recover within 256. Timing room is therefore not sufficient across these saved cases. The periodic and triggered schedules are different treatments, not a clean causal intervention on exploration length or an SNN benefit.

## Dependency retention is not the same as accuracy

Initial pools plus every saved `added`/`removed` set exactly reconstruct all **100 final pools**. Checks reject repeated/overlapping starts, invalid cooldowns, mismatched 180-receipt commits, foreign/duplicate route IDs, incorrect pool sizes and unfinished-window flags. A window ending at state 1024 has no subsequent begin operation and is not incorrectly credited with a commit. No pool-bookkeeping inconsistency is found **within the saved metadata**; that does not validate unrecorded execution details.

| Registered exact-retention failures | Reported tail / frozen accuracy | Interpretation |
| --- | --- | --- |
| Independent, seed 1237 | 100% / 100% | Exact route-membership failure with high finite predictive accuracy |
| Interacting, seed 1237 | 100% / 100% | Same distinction; the retention gate still fails |
| Interacting, seed 1201 | 45.31% / 54.30% | Retention failure with poor reported accuracy |

The two high-accuracy cases do not prove that a substitute route is functionally equivalent. The true missing route IDs, per-route score margins, recent support counts and local errors were not saved. A state digest cannot supply them. Candidate IDs are audited only as opaque set members; the generator and seed permutations are not reconstructed to manufacture missing evidence.

The following remain explicitly **unknown**: which exact dependencies are missing; whether support, stale counts, ranking cost or another mechanism explains selection; whether a retained superset substitutes causally; the within-block prediction order; and whether shorter exploration would improve recovery without harming quality. This audit neither rescues exact-retention failures nor attributes them to a specific unobserved learning defect.

## Verification and next evidence boundary

**90 tests pass under Python 3.10.22**: 34 new report/arithmetic checks and 56 retained pure v2-comparison/codec/path checks. They cover sharp count bounds, mixed qualifying blocks, periodic timing versus quality, high-accuracy retention failures, all five arm lifecycles, final-boundary commits, malformed starts/pools, integer block counts, strict metadata types, proof/hash/negative-result mutations and consumed-attempt refusal. The test process imports no learner. The retained audit result exactly reproduces in a fresh process.

| Artifact | SHA-256 |
| --- | --- |
| Selection | `c3a740c6049bd2b9e3f2a7facf63d2904e3c71b92f54249727dc9dcd43dcfeed` |
| Code registration | `4f502cc9e414ef2838ffb3ed7f8f89ca9b8822cc5dd9853924c49a55712c5b52` |
| [Audit result](../data/processed/benchmark_fixtures/route_failure_audit_result_v1.json) | `7edeb990878d92029c5be7d54254982f9f09df30717e2ab3c2ef5262458c1595` |
| [Pure replay](../data/processed/benchmark_fixtures/route_failure_audit_replay_v1.json) | `e265d7a88646241e9a486ffc84af01b8627cb95f16992147ae00d2953989613e` |

```sh
# Saved-report operations only; no learner or generator execution.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/route_failure_audit.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_route_failure_audit.py tests/test_route_numeric_v2.py tests/test_reactivation_codec.py tests/test_project_paths.py
```

⭕️ [Pending] Causal attribution cannot be completed from these reports alone. A separately authorized future protocol would need bounded, provenance-linked route identity/support/score snapshots at already defined boundaries to distinguish missing support from ranking/retention effects. Such evidence must be specified before any trial; no snapshots, learners, new scoring, parameter changes or resource diagnostics are added here. Keep the original recovery/evidence/current-state gates and all negative/sealed outcomes. No further automatic audit or numerical run is queued.

Selective circuit use remains a design opportunity shared with non-spiking controls. Timing arithmetic and retained pool IDs do not establish an ANN-absent SNN advantage, independent usefulness or energy savings. Existing R2/task/resource/production holds remain in force.

日本語: 保存済み7報告・1,593,723バイトだけを使う有限監査を実装前に登録し、90テストと別プロセスの完全再現が成功しました。再探索開始6〜13tick、確定186〜193tick、期限残り63〜70tickを切り分け、全100候補群の増減記録を照合しました。確定後2ブロックが収まらないことと、元の基準が不可能だったという主張は区別します。依存保持失敗3件のうち2件は報告精度100%ですが、保持判定は不合格のままです。真の欠落依存・候補別スコア・支持不足の因果原因は未記録で未知とし、432既存ハッシュ・旧再現失敗・負の結果・封印を保持しました。学習器再実行・採点・条件調整はゼロです。

简体中文：在实现前注册仅使用7份保存报告、1,593,723字节的有限审计，90项测试与独立进程完全重放通过。分离触发延迟6–13 tick、提交延迟186–193 tick及期限剩余63–70 tick，核对全部100个候选池的增删记录。提交后两个完整区块无法容纳，不等于原门槛不可能达成。3项精确依赖保留失败中2项报告准确率为100%，仍保留失败判定。缺失依赖、候选分数与支持不足的因果原因未记录，保持未知。保留432个既有哈希、旧重放失败、负结果和封存，不重跑学习器、不评分、不调整条件。

🟢 [Done] Follow-up design: [prospective bounded route snapshot collector/contract](SPARSE_ROUTE_SNAPSHOT_CONTRACT.md) implements the missing evidence interface and static capacity checks. Actual hook integration/collection remains pending; no old state is reconstructed or learner restarted.
