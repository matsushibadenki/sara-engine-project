# Current-relative local evidence prerequisite — 2026-10-15

This component study follows the negative HH103 support-8 result. It uses fresh household HH104 and fixes the rule before reading that household's sensor/message fields. It does not authorize opening independent development or test periods.

The source is the [official CASAS longitudinal archive](https://zenodo.org/records/15708568), member `data/hh104.csv`. Household identity is fixed by the smallest unused HH identifier, not selected by task labels. Only the ZIP directory name, byte offsets, sizes and CRC were inspected before registration. HTTP range acquisition avoids downloading or opening other household members. Full-archive MD5 is published by the provider but is not claimed as verified; member compressed SHA, expanded SHA, CRC and byte length are checked locally.

## Frozen hypothesis and controls

For one joint history branch, let D be its two outcome counts. Subtract D from the counts for its current-state context to obtain disjoint complement C. With n=sum(D), m=sum(C), activate the joint posterior only if both are nonzero and:

```
abs(D_positive/n - C_positive/m)
    > sqrt(log(80)/(2*n)) + sqrt(log(80)/(2*m))
```

Otherwise use precisely the current reference's global → current → current posterior, with prior strength 8 at each step. This fixed Hoeffding separation proxy requires evidence beyond support alone. The records are serially dependent: the rule is not a calibrated 95% significance test or a repeated-testing guarantee. No threshold search is allowed.

All eleven arms are registered together: current, finer current (five-minute clock bin), history histogram, transition, joint, previous support-8 policy, evidence candidate, ungated constant-gap, evidence with constant gaps, shuffled-feedback evidence and scalar evidence duplicate. The scalar duplicate uses the same scalar implementation; equality verifies replay integrity and establishes no spike-specific advantage.

Each due feedback increments exactly three local count cells, including the detail cell when the gate is inactive. Current inputs remain available to every arm. Branch activity counters are reporting-only. Frozen confirmation changes zero learned counts. The bounded queue delivers feedback in due order before later predictions; no gradient, matrix, GPU or whole-model scan is required.

Outer dates use the unchanged metadata splitter. Inside the outer training dates, the first floor(2N/3) dates fit and the remainder provide internal confirmation. Original independent development and test are discarded by date before sensor/message or time interpretation during extraction. Timestamp/schema metadata is inspected across the source to establish chronological dates and acquisition integrity; sealing refers to predictive fields and outcomes.

All previous source gates remain mandatory, including 180 recorded chronological dates, both classes (fit >=500/class on >=20 days; confirmation >=200/class on >=10 days), repeated current states and >=10 opposing-history contrast groups with >=4 observations per opposing label. Failure stops before model scores. On qualified data, all quality gates must pass: >=.01 mean daily Brier gain and >=.02 balanced accuracy gain over the best of all five sufficient references, >=.005 constant-gap loss, >=.01 shuffled-feedback loss and exact scalar equivalence. No source substitution, relaxed threshold, retuning or test opening follows failure.

## Artifacts and reproduction

The immutable protocol, registration, acquisition, timestamp metadata, integrity receipt, consumed execution marker and result live under `data/processed/benchmark_fixtures/casas_evidence_*_v1.json`. Registration pins code and earlier JSON/gzip evidence. Readouts, if source qualification permits scoring, live only under `models/casas_evidence/`.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/casas_evidence_diagnostic.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_casas_evidence.py
```

Replay requires the pinned raw member in `data/raw/r2_casas_evidence/hh104.deflate`. Raw bytes and response headers are ignored by Git. `register`, `metadata` and `execute` use exclusive artifact creation, so consumed identities cannot silently overwrite the original run. Replay checks preserved evidence, pinned inputs, exact refit results and saved checkpoints.

## Retained result

🟢 [Done] Protocol `a344d877ede84fe256177dd6552f463da52b04bbdbe498dcc33bdaa398fbb0f4` and implementation were registered before new household values. The source has 4,941,414 records on 812 chronological recorded dates. Its outer split is 180 training / 60 independent development / 572 test dates. Internal fitting is 2011-06-15 through 2011-10-12 (120 recorded days), confirmation 2011-10-13 through 2011-12-11 (60 days). Both independent periods from 2011-12-12 onward remain predictive-field sealed: **3,845,589 records**.

All source conditions pass: fitting has 6,488 positives / 20,866 negatives on all 120 days; internal confirmation has 3,616 positives / 11,936 negatives on all 60 days. All 19 observed current states recur; 1,104 rich contexts have at least eight observations; 101 opposing-history groups exceed the required ten. There are 8,578 late outcomes, which are excluded rather than made negative, and three unresolved outcomes. Shared extraction and queue limits are unchanged.

Every arm executes **82,062 changed local writes** (27,354 fit examples × three), zero confirmation updates and at most five pending prediction receipts. Saved-state prediction replay and fresh-process refit replay are exact; 104 related tests pass. The replay receipt pins the result and registration hashes. The 253 earlier JSON/gzip evidence artifacts remain hash-identical.

| Arm | Mean daily Brier ↓ | Balanced accuracy ↑ | Confirmation overrides |
| --- | ---: | ---: | ---: |
| Current | 0.150693005 | 59.5286% | — |
| Finer current | 0.152081296 | 58.1146% | — |
| Histogram | **0.150623493** | 59.4153% | — |
| Transition | 0.150890194 | 59.3529% | — |
| Joint | 0.151854963 | 57.2882% | — |
| Previous support-8 | 0.150859197 | 59.5164% | 998 |
| Evidence | 0.150693005 | **59.5286%** | **0 / 15,552** |
| Constant-gap joint | 0.150890194 | 59.3529% | — |
| Constant-gap evidence | 0.150693005 | 59.5286% | 0 |
| Shuffled-feedback evidence | 0.197696105 | 49.1050% | 0 |
| Scalar evidence duplicate | 0.150693005 | 59.5286% | 0 |

**The hypothesis fails.** Brier gain is -0.000069512 rather than >=0.01; balanced gain and constant-gap loss are both zero. Shuffled-feedback loss is 0.047003100 and scalar equivalence passes, but the former is fully explained by corrupted current-state feedback because the gate never activates. Neither proves a temporal mechanism. Zero overrides establish no incremental use by this fixed rule on this fitted representation, not absence of temporal information in the source. This is training-only evidence, not independent validation, a runtime improvement, or an SNN advantage.

🟠 [Next] Before acquiring another household, preregister a training-only feasibility diagnostic for bounded pooling of temporal histories. Preserve current context and distinguish histories with opposite futures; verify usable disjoint evidence coverage under a fixed state/write budget before spending fresh identities on another comparison. Do not weaken the completed gate or search thresholds on HH104 confirmation.

🔴 [Later] Independent validation, factor growth and runtime integration require a separately registered positive component prerequisite. All previous negative results and seals remain in force.

日本語: 未使用HH104の成立条件はすべて通過し、11群の実更新を各82,062回に揃えました。しかし新ゲートの有効化は0件で、現在状態対照と予測が完全一致しました。改善・時間情報の必要性は不合格です。既存の負の結果と3,845,589件の封印を保持し、閾値は変更しません。次は新世帯を使う前に、訓練期間だけで時間履歴の限定的な集約が十分な比較根拠を残せるかを事前登録します。

简体中文：未使用的HH104满足全部数据条件，11组各执行82,062次实际局部更新。但新门控启用0次，预测与当前状态对照完全相同，改进与时间必要性条件未通过。保留既有负结果及3,845,589条封存记录，不调整阈值。下一步在消耗新家庭之前，预注册仅训练期的有界时间历史聚合可行性诊断。
