# Fixed-condition real route-evidence capture v3 — 2026-10-27

🟢 [Done] The separately preregistered v3 connects the unchanged [read-only collector](SPARSE_ROUTE_SNAPSHOT_CONTRACT.md) to actual models. **All 160 registered boundaries are retained, and capture/no-capture numerical payloads match exactly.** All 160 individual captures preserve complete model-state hashes. Newly captured evidence separates the existing support threshold, score ranking and exact dependency retention. The scientific component remains negative; no old execution is resumed or reclassified.

## Registration, controls and method isolation

The [v3 selection](../data/processed/benchmark_fixtures/route_capture_v3_selection_v1.json) was saved at **20:04:13 JST**, before implementation. [Code/tests registration](../data/processed/benchmark_fixtures/route_capture_v3_registration_v1.json) at **20:12:49 JST** preceded the new uninstrumented baseline at **20:12:50 JST** and capture execution at **20:13:02 JST** in a different process. The owner's explicit request authorizes this separately versioned actual collection. Scientific conditions and the exact CPython **3.10.22** runtime remain unchanged.

Each phase covers the original two tasks × five seeds × stationary/changed conditions: **20 cases, five arms per case, 1,024 delayed learning receipts per arm**. Every arm/case still performs 9,216 actual route updates and 36,864 changes of the four registered learning fields. All original seeds, generation, score/readout rules, threshold 0.20, 180-tick exploration, cooldown, candidate limits and scientific gates are copied and hash-checked. The original learner, old runner, codec and collector source files remain immutable.

The [external hook adapter](../src/sara_engine/evaluation/route_capture_v3.py) temporarily wraps only the existing class's `narrow` and `observe` methods inside the new capture context, invoking each original method exactly once. It adds **no instance fields**. Non-target arms execute the originals directly. Actual shortlist/start/commit transitions must match the registered sampling plan; unexpected or missing events fail rather than creating unregistered records. Every method is restored on normal exit or exception.

The [new runner](../scripts/eval/route_capture_v3.py) never calls an old execution/replay command. Baseline/capture phases use exclusive new namespace markers, fixed storage ceilings and saved-result gates before any learner import. The baseline must be retained and agree exactly with the historical reference before collection can start. Full numeric reports and the complete collection are retained before analysis/final proof; failures stop without retries or capacity changes.

## Numerical noninterference and complete evidence

The unchanged `ReportCodec` compares all scientific fields, all 100 model-state digests and all 100 prediction/update ledgers per phase. It preserves types, ordering and exact floats. Historical-reference comparison uses only the previously registered two provenance bindings (`registration_sha256` and `preserved_artifacts`, now 450 pins); baseline and capture payloads then compare in full with no additional exclusions. Both also match the historical scientific report under that same explicit mapping.

| Evidence | Verified result |
| --- | --- |
| Actual model snapshots | **160 / 160**, all 45 candidates per snapshot |
| Pre/post initial shortlist | 40 pairs / 80 snapshots |
| Pre/post exploration start | 10 pairs / 20 snapshots |
| Pre/post completed commit | 10 pairs / 20 snapshots |
| Final observed states | 40 snapshots |
| Individual capture before/after full model digest | **160 unchanged** |
| Initial/start/commit paired learner rows | All 60 pairs unchanged; intended scheduling metadata may change |
| Final captured full-state digest | All 40 link to the numeric reports' final model digests |
| Numerical baseline/capture payload | Exact equality |

The original method order trains the permanent control before the triggered arm, while the registered archive groups triggered records first. Every snapshot is keyed by its full registered identity and serialized in that declared archive order. The separately saved actual arrival order and all per-capture before/after hashes retain provenance; no record is dropped or silently conflated.

True dependency IDs are produced by the pinned evaluator **after all phase learner work and hook restoration**. They are not passed to the learner or `capture`. This is newly collected v3 evidence, not a reconstruction of missing old states. The contract tag `future_external_state` now identifies externally captured actual state, bound to this new code registration and capture marker.

## Support, ranking and exact retention

All 30 true dependencies inspected at the ten changed commits meet the unchanged **16 total observations** readout requirement; cumulative observations range **72–412**. Every route receives exactly **36 observations** during each 180-tick exploration. All true-route key supports in those windows are nonzero, ranging **6–24** per key. There is no new per-key statistical-sufficiency gate: these descriptive counts do not prove that support is adequate for every causal or calibration claim.

The three exact-retention failures now have explicit evidence:

| Case | Missing route ID | Observations at commit | Rank at commit | Cost above third place | Window key supports |
| --- | ---: | ---: | ---: | ---: | --- |
| Independent seed 1237 | 3 | 72 | 4 | 0.0575263681 | 19, 17 |
| Interacting seed 1201 | 32 | 412 | 8 | 0.0191551648 | 8, 10, 12, 6 |
| Interacting seed 1237 | 3 | 72 | 4 | 0.0575263681 | 19, 17 |

These are **score exclusions, not ID tie-breaks or failures of the existing total-support threshold**. Post-commit pools match the fixed top-three-per-output rule. The missing routes remain absent at the final state; their final ranks are 4, 5 and 4 respectively. Snapshot/evaluator membership exactly matches the unchanged numerical report's retention flags and initial-new-dependency contrast flags.

A diagnostic ranking from the captured **window residual/weight differences** also leaves all three missing routes outside its top three. This is not evidence that replacing a cumulative score with a recent-window score would repair the task. Window residuals were produced by predictions from the original cumulative count tables; they are not the predictions of a reset/windowed learner. No alternative rule is applied or scored.

The two seed-1237 cases retain 100% tail/frozen accuracy while failing exact route retention. The interacting seed-1201 case remains at 45.31% tail / 54.30% frozen accuracy. Accurate finite predictions do not establish functional substitution, and high observed error does not identify why the current score became unfavorable. Historical count contamination, probability adaptation, width cost and other causal explanations are **not isolated** by observing state. Causal attribution, reset/windowed counterfactual accuracy and functional-substitution proof remain explicitly unknown.

All ten changed cases still fail the unchanged 256-tick recovery criterion. The earlier timing decomposition and the old consumed replay failure remain intact. No threshold, deadline, seed, retained-dependency rule or outcome definition is adjusted.

## Artifacts, tests and bounds

Baseline/capture numeric reports occupy **499,867 / 499,866 bytes**. The compact [160-snapshot collection](../data/processed/benchmark_fixtures/route_capture_v3_collection_v1.json) is **399,362 bytes**, with a maximum individual record of **2,699 bytes**, below the contract's 8,192-byte record / 2 MiB report ceilings. The [analysis and arrival/hash audit](../data/processed/benchmark_fixtures/route_capture_v3_analysis_v1.json) occupies **250,549 bytes**. Nine artifacts total **1,744,840 bytes**, including **345,745 auxiliary bytes**, below the newly declared 7 MiB total / 1 MiB auxiliary ceilings. Scientific numeric-report and collection limits remain 2 MiB each. All **450 prior file hashes** and registered sources verify.

**143 tests pass** in two isolated processes: 16 new tests, including a preregistered development-only 192-tick hook/no-hook comparison on seed 81001, plus 127 retained pure contract/audit/comparison/path tests. New tests cover exception-safe restoration, complete-state mutation detection, unplanned events, missing/consumed baseline gates, fresh-process collection, numerical drift and completed-evidence verification. The development witness is not a study seed or effectiveness result. Separating test processes preserves the pure suite's no-learner-import check. Final verification recomputes the proof from saved evidence without learning or evaluator seed regeneration.

| Artifact | SHA-256 |
| --- | --- |
| v3 selection | `f6bbe9940c42d47ced0b846e6de2406b28f91aaae9592c8f558515eb7450d68e` |
| Code registration | `9b22b7e731714ea3bb80ef7df432da54d7b2caa834ab935f3f7ed41703e64112` |
| Baseline numeric report | `6fbddcd1bf569d9cfae4955708158ab85259cf3bc4a8de97bce7f1ce5f3b27e0` |
| Capture numeric report | `38dfcfb305ccd3540d6d7e26b68218a728a446f4e168bec392383deb4a0813e4` |
| Collection | `248c6ca8dc53b6f0568044ce259298f4275257f26900c55bf4dad18a912978ab` |
| Analysis | `cfc2b4404342dc70f5602910f8daff7d7afc72c182f69504ff4ed375fcec15fa` |
| [Final proof](../data/processed/benchmark_fixtures/route_capture_v3_result_v1.json) | `242ad2b06b634339be3970aa4dbcdc6cd663c33c12da316c5cc22b278bdf0686` |

```sh
# Completed evidence verification, no study restart.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/route_capture_v3.py verify
# Run integration and pure checks in separate processes.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_route_capture_v3.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_route_snapshot_contract.py tests/test_route_failure_audit.py tests/test_route_numeric_v2.py tests/test_reactivation_codec.py tests/test_project_paths.py
```

⭕️ [Pending] Any rule change or causal intervention requires a new, separately justified protocol. This evidence does not authorize retuning or another collection run. Observer hashing/encoding adds work; numerical noninterference is not zero overhead or energy superiority. Independent task qualification, branch-local causal credit, nonstationary promotion, physical-energy/resource work and production remain under existing holds. The scalar/non-spiking setting establishes no ANN-absent SNN capability. Old failed replay metadata and all sealed periods remain unchanged.

日本語: 条件を固定した別登録v3で採取器を実モデルに接続し、採取なし基準と全数値の完全一致、160境界の完全性、採取前後の全状態ハッシュ不変を確認しました。依存保持失敗3件は支持数72／412／72で16観測基準を満たし、確定時4／8／4位のスコア除外でした。同率タイブレークや既存支持閾値不足とは切り分けましたが、スコア不利の因果原因・規則変更の反実仮想精度は未知です。143テストと保存証跡検証が成功し、450既存ハッシュ、旧再現失敗、負の結果、条件、封印を保持しました。

简体中文：在条件固定、独立注册的v3中，将采集器接入实际模型，确认与无采集基准完整数值一致、160个边界齐全、采集前后完整状态哈希不变。3项依赖保留失败的支持数为72／412／72，均满足原16次观测门槛；提交时排名4／8／4，均因分数被排除，而非同分决胜。分数不利的因果原因和规则更改后的反事实准确率仍未知。143项测试和保存证据验证通过，保留450个既有文件哈希、旧重放失败、负结果、原条件及封存。
