# Discovery Capture Pair: Development Result

Protocol: [`discovery_capture_pair_v1.json`](../data/processed/benchmark_fixtures/discovery_capture_pair_v1.json), SHA-256 `dbe0d1999a0af71ff4d058cc484c81d0bcdc96a00ee6425a899f5ae8b3043221`. Candidate source SHA-256: `8fbfcc8efb8f19773c789d374b670d0553ecea6acb56aa492d2f8742dd8d76e0`. Capture: `workspace/evaluation/discovery_capture_pair_v1_20260917/capture.jsonl`, file SHA-256 `64cb647e19dfe54b855b72441089314aa8c945ccabfcfde064522fb58d5bad24`; final chain head: `d9ce4224b1472b5d21480b406d3438348b5c04ad5685e4b13c7987e00ee31075`. The complete-log projection is **unapproved**, SHA-256 `5107373cb10a61cf17ca7917532738e487936f038ea08da1cc0f7779365ae133`.

Each `prepare` returned before evaluation, and its intent head was recorded in the conversation before a separate `execute` call. This is a visible chronological record, not an independently authenticated external timestamp or human approval.

| Fixed action | Pre-evaluation intent head | Accuracy / 250 development predictions | Process CPU ms | Input events | State bytes | Maximum event work | Prediction trace SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| B compact event | `031769437535cdbe1748d2241bc5a5baa60e33fc3384eac59c0ff773bf720ffa` | 0.74 | 11 | 1,800 | 3,034 | 3 | `a093684a7621529cc9b17e2a182c3593a72c1ada45b32d5b2d0c2ee6f334fcbd` |
| C temporal state | `a07873a27539c2d9ad4a3f1d5f0dc17adcc8f90d9c24837df585a294018b17f1` | 0.94 | 12 | 1,800 | 8,939 | 11 | `5b30b602f8169524d8d3edecad5605a0a19b752d345ee5b325764dce4988888f` |
| C time shuffle | `c18bdfdb7970cc94944fe09ecde0e168dcfbc5033a2443870a0538bb0b08d38b` | 0.84 | 12 | 1,800 | 7,391 | 11 | `d1188d15b15c99d30f3a80ace5cfba2d856960b98ca726a77ded01cac0411742` |
| C state reset | `e38ae4ea83332fb6660e9fe5bb5e4eef91125a55b471a01e682daec744347758` | 0.74 | 12 | 1,800 | 3,034 | 9 | `a093684a7621529cc9b17e2a182c3593a72c1ada45b32d5b2d0c2ee6f334fcbd` |

The read-only verifier replayed the fixed four actions using the frozen source and compared all four exact prediction-trace digests, accuracy, retained state, input-event count, and final capture head. All preregistered diagnostic checks passed: C−B = +0.20 (minimum +0.10), all five per-seed C−B differences are positive (+0.20, +0.18, +0.20, +0.28, +0.14), C−time-shuffle = +0.10 (minimum +0.05), C−state-reset = +0.20 (minimum +0.10), and retained state/event work are under their fixed limits. The five development seeds are 940001, 940113, 940227, 940341, and 940457.

The read-only lineage audit matched the four caller-supplied intent heads, capture-file digest, final chain head, and projected snapshot digest. It found **zero overlapping episode IDs but 250/250 development examples with an identical `(family, input events, label)` pattern in training**; only 28 unique labeled training patterns exist. The apparent seed split therefore provides no exact-pattern isolation. This audit does not independently authenticate the conversation transcript or review the original research history.

Interpretation: on this same synthetic task family, local temporal state improves development accuracy under the fixed evaluator, but the 100% exact-pattern overlap prevents a generalization claim. This is a replication of an already known contrast, **not** independent task transfer, spike-specific gain, measured energy efficiency, or exploration-policy improvement. C uses more retained state and event work than B; measured process CPU is too small and noisy to support an efficiency conclusion. No BPI/Sepsis frozen test, prior event-unit held-out set, or production gate was opened. Because source-pattern isolation failed and independent head attestation is absent, this capture must **not** be exported as an approved discovery tree. Preserve it as a diagnostic result and preregister a genuinely pattern-disjoint task before further causal or policy claims.

日本語: 固定した開発実験では時間状態CがBを0.20上回りました。しかし開発250例すべての入力・正解パターンが学習側にもあり、汎化検証にはなっていません。スパイク固有の利得や効率改善も示していません。このcaptureは承認済みtreeへexportせず、診断結果として残します。

简体中文：固定开发实验中时间状态C比B高0.20，但全部250个开发样本的输入与标签模式都已出现在训练集中，因此不能证明泛化，也不能证明脉冲特有收益或效率改善。此采集仅保留为诊断结果，不导出为获批研究树。
