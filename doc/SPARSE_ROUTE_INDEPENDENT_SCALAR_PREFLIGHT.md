# Independent stateless scalar arithmetic preflight v1

🟢 [Done] The separately preregistered, nonlearning check passes all 11 existing hand records: 22 exact value comparisons, 11 detected faults, 12 unit tests and identical fresh-process replay. This closes the fixture-arithmetic gap selected by the saved-proof readiness audit, within this scope only.

## Frozen protocol and independence

The selection was saved before source registration and execution. Its SHA-256 is `eeecf9fb7bb6ebbf9f34f7013faabe9e43b179f16eb2078c0f477832c2514b52`. The only parsed input is `route_error_binding_preflight_selection_v1.json`, `/local_rows`: 78,637 bytes, with the 11 rows fixed by canonical SHA-256 `6b577f8f687212ab1417d72b841f979f18b7a251f295e7fca1002258a2719953`. No rows or expected values were regenerated or selected after execution.

The new kernel `src/sara_engine/evaluation/route_independent_scalar.py` imports only `math.isfinite`. Its pure `calculate(tick, history, error_weight, allocated, n0, n1, side)` function receives no golden values, invokes no existing calculation helper, maintains no learner/global state and returns a fresh dictionary. The harness alone reads expectations. A source AST guard rejects helper imports, mutable global state, dynamic calls and default arguments; a module guard rejects loaded primary calculator/learner modules. These are scope checks, not security attestation or authenticated runtime provenance.

For public scale `s = 2**(tick//64)`, prior is `s` under discounted history and 1 under cumulative history. Allocated probability is `(n1 + prior) / ((n0 + n1) + 2*prior)`. Unallocated keys retain `None`, with residual fallback 0.5; allocated zero-evidence keys return 0.5. Error increment is the squared difference from the credited binary side, multiplied by either 1.0 or the public scale. This recomputes only the recorded local arithmetic, without modifying a count table or selecting a route.

## Verification and bounded execution

Exact comparison requires float type, finiteness and identical `float.hex()`; the absent probability requires `None`. There is no tolerance. All 11 probability comparisons (including one absent value) and 11 increment comparisons match. The primary and fresh-process replay each calculate exactly 11 records. Verification recomputes the same fixed arithmetic; it does not execute a learner. The 12 tests also cover fresh returns, unchanged module state, asymmetric counts, unallocated versus allocated-zero keys, invalid input types and exact call counts.

All 11 preregistered faults are detected: one-ULP probability and increment changes; missing seen and invented absent probabilities; integer-for-float probability and increment; changed row digest; primary-helper import; mutable global; dynamic call; and default argument. Fault checks use literal copies and source text, rather than extra kernel calculations. A mismatch is retained and fails closed without formula changes or retry.

Limits remain one input of at most 262,144 bytes, 11 records / 22 comparisons per phase, 262,144 bytes per artifact and 2,097,152 bytes aggregate. Source/runtime registration pins CPython 3.10.22 and the new kernel, harness, runner, tests and path utility. Verification preserves 539 historical hashes and the absence of the old failed replay's success artifact. Consumed execution markers prevent primary/replay restart.

All six artifacts are under `data/processed/benchmark_fixtures/`, with prefix `route_independent_scalar_preflight_` and suffix `_v1.json`: selection, registration, execution, result, replay_execution and replay. The result retains each input, literal expectation, actual value, hexadecimal representation, comparison and fault outcome.

## Interpretation and next step

Actual learning updates, learner executions, task prediction scoring, acquisition and resource measurements are all zero. Old replay failure, negative recovery/retention findings, unknown causal causes, unchanged decision thresholds and all test/source seals remain preserved. Matching these hand records does not establish independent runtime learner equivalence, genuine-write provenance, independent task usefulness, adaptive-system acceptance, or SNN-specific accuracy/energy benefit. Additional resource measurements and promotion remain on hold.

🟠 [Next] Use saved evidence only to specify the minimum nonlearning independent scalar state-transition contract, distinguishing arithmetic equality from genuine runtime writes, delayed-outcome provenance and full learner equivalence. Preregister scope and fixed before/after expectations before any new implementation or execution. No additional learning, task scoring, acquisition, resource measurement or condition adjustment is authorized by this result.

## 日本語

既存11件・22値は型と `float.hex()` が完全一致し、故障注入11件、単体テスト12件、別プロセスでの再検証も成功しました。期待値を受け取らない独立・無状態関数で計算し、既存ヘルパーは使用していません。539件の既存ハッシュ、旧失敗・負の結果・封印を維持しています。確認できたのは手書き記録の算術だけです。実モデルの独立同等性・実更新の由来・SNN固有の優位性は未確定で、追加学習・採点・取得・資源計測は行っていません。次は保存済み証拠だけで、非学習の独立状態遷移契約に必要な最小条件を定めます。

## 简体中文

既有11条手写记录的22个值在类型和 `float.hex()` 上完全一致；11项故障注入、12项单元测试及独立进程复验均通过。独立无状态函数不接收期望值，也不复用已有计算助手。保留539项历史哈希、旧失败、负面结果及封存条件。本次仅验证固定记录的算术，未证明实际学习器的独立等价性、真实更新来源或SNN独有优势；未新增学习、任务评分、采集或资源测量。下一步仅依据已保存证据界定非学习独立状态转移契约的最小要求。
