# Branch-local delayed-credit design contract — 2026-10-25

🟢 [Done] Implemented an executable **static proposal contract** for the conditional branch-local delayed-eligibility design selected by the [retained-evidence decision](R2_MECHANISM_DECISION_GATE.md). This closes a prerequisite design gap: update-budget equality must not be manufactured with no-ops, repeated writes or missing controls. It does not implement a learner or establish causal/SNN usefulness. Experiments and additional resource work remain held.

## Registration and scope

[Selection](../data/processed/benchmark_fixtures/local_credit_contract_selection_v1.json) was saved at **17:15:28 JST**, before implementation; [code/tests registration](../data/processed/benchmark_fixtures/local_credit_contract_registration_v1.json) at **17:17:21 JST** preceded the retained static check at **17:17:34 JST**. The finite design contract freezes six arms, a symbolic metadata witness and rejection rules. Five retained artifacts total **7,997 bytes**, each below the 65,536-byte artifact ceiling. It pins the six prior decision artifacts and verifies their inherited evidence/code pins, including 386 prior artifacts. Existing raw streams, models, sealed periods, failed attempts and registered experiments are not changed or rerun.

| Artifact | SHA-256 |
| --- | --- |
| Selection | `26c42e680d62acce6be1f3619952dfe647392e7ee9d74890eb836f31621eafd6` |
| Code registration | `376e3be661195847a6233697dde468fe07e226c3ee992c33ed8e7eef32376e22` |
| [Static result](../data/processed/benchmark_fixtures/local_credit_contract_result_v1.json) | `6ec73221532176ed40c42d6f83a08bf7a68e1116aa07d2ec0ae320e1adb841b0` |
| [Exact replay](../data/processed/benchmark_fixtures/local_credit_contract_replay_v1.json) | `f7c63c8ec10bbd4853980f804fc14b975c7eefb018590c7f8f6a46824eef6daf` |

## Implemented boundary

The [pure validator](../src/sara_engine/evaluation/local_credit_contract.py) takes one proposed receipt, bounded already-consumed receipt IDs, shared event metadata and six arm-local proposed updates. It never applies updates, consumes a receipt, predicts or scores an outcome. Its inputs are assertions requiring independent executor provenance in any future experiment.

- All six arms are required: strong current, sufficient history, intact eligibility, disabled eligibility, permuted eligibility and an equation-matched scalar emulator. Shared observer digests must match, and declared state ceilings must be equal. This verifies supplied metadata equality, not actual observer execution.
- The observed outcome must lie within the original inclusive 300–600-second window; delivery must follow observation and precede or equal expiry, within the same 600-second bound. Only delivered results qualify. Missing/cancelled statuses, late deliveries and already-consumed receipt IDs are rejected. Receipt reuse is checked against the supplied bounded ID set, not against an implemented persistent ledger.
- Shared history contains at most eight event IDs, each no more than 300 seconds before receipt creation. Updated routes must exist in that arm's bounded state, and cited events must belong to their declared causal membership. This does not verify that a branch was actually active or that an ablation was actually performed.
- Each atomic route proposal changes a bounded integer weight and increments its observation counter exactly once. Both proposed changed fields are counted. Old values must match supplied state. No-op weight proposals, counter-only padding, repeated routes, private events and foreign receipts fail. A zero-update ablation cannot be padded into parity.
- Every arm must propose the same positive changed-field count, at most 16 per receipt. The symbolic witness has two route proposals, hence **four proposed changed fields per arm, zero applied writes**. Identical intact/emulator state and proposals are required, rather than a claim of superiority over an equivalent equation.

The 4,096 declared-slot ceiling and bounded integer representation are design limits for this validator, not new empirical state measurements or substitutes for original task budgets. Declared slots must at least contain supplied route values; metadata, receipt/history storage, aliasing, complete executor state and temporary allocation are not measured or certified. Mathematical learning-rule correctness is not checked: the tests deliberately demonstrate that another bounded changed weight can satisfy the structure without proving a valid learning equation. Future work must separately freeze that rule and prove actual positive changed-write/state fairness without weakening existing gates. General learning updates that change only counters or reach saturation are not certified by this deliberately narrower candidate contract.

The [runner](../scripts/eval/local_credit_contract.py) has exclusive registration/execution files, verifies source and prior evidence hashes, and supports exact pure replay. It parses only the already-retained decision aggregate to copy unresolved prerequisites. It has no experiment, tracer, acquisition or scoring command. A passing symbolic witness never authorizes execution or resource measurement.

## Verification and decision

**100 tests pass under Python 3.10.22**, including 41 new contract tests and 59 retained pure decision/path tests. Counterexamples cover causal time bounds, inclusive expiry, absent controls, receipt reuse, counter padding, duplicate writes, no-ops, stale state, private history, state bounds, boolean-as-integer coercion, observer mismatch and emulator disagreement. Validation preserves the proposal without mutation. A separate-process static replay exactly reproduces the retained result. The historical resource-runtime tests are not rerun or changed.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/local_credit_contract.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_local_credit_contract.py tests/test_r2_mechanism_decision.py tests/test_r2_research_decision.py tests/test_project_paths.py
```

⭕️ [Pending] Independent task qualification and causal branch-local credit effects remain unestablished. The prior aggregate's false independent-usefulness result and unknown candidate prerequisites remain unchanged. Preserve all exact-current, opposing-history, class/support, delay, 180-date/270-for-180-fit, 70,212-arrival and sealed-test conditions. New prediction experiments, source acquisition/search, resources, automatic diagnostic v3 and production integration remain closed until a separately qualifying protocol exists. No additional static-audit loop is queued merely to restate this hold.

The circuit-optimization idea motivates testing selective local credit, but sparse activity and event scheduling are shared by scalar/ANN-style implementations. These contract checks establish neither an ANN-absent capability nor energy savings. A future mechanism needs causal ablation evidence and independent usefulness before resource comparisons can inform a decision.

日本語: 枝ごとの遅延適格性について、実行前の更新案を検査する静的契約を事前登録・実装しました。未来・期限外・空更新・重複・水増し・対照不足を拒否し、関連100テストと完全再現が成功しました。各方式4件は提案上の変更フィールド数であり、実際の学習更新はゼロです。因果効果・独立課題・実行時の公平性は未実証のため、追加実験・資源計測を保留し、既存条件・負の結果・封印を保持します。

简体中文：预注册并实现分支局部延迟资格迹的静态更新提案契约，拒绝未来或过期事件、空更新、重复、虚增计数和缺失对照。100项相关测试及完全重放通过。每种方法的4项仅为提案中的变化字段，实际学习更新为零。因果效果、独立任务和运行时公平性尚未确立，继续暂停额外实验与资源测量，保留原条件、负结果和封存。
