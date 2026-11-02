# Independent scalar state-transition contract v1

🟢 [Done] A separately preregistered, saved-evidence-only contract fixes the minimum next independent transition check. Six unit tests, five rejected contract/input mutations and exact fresh-process static replay pass. All 545 historical hashes verify. This phase executes no transition, calculator, learner, task scorer or resource measurement.

## Fixed sources and registration

Selection SHA-256: `b79bdd0f95bcb03f7fbdf90e85a85c095d43ba172eb8bca0c4ebd28c7f0716c7`. The selection preceded implementation; source, tests, path utility and CPython 3.10.22 were registered before static execution. Only two saved inputs are parsed: `route_numeric_delta_preflight_selection_v1.json` and `route_independent_scalar_preflight_result_v1.json`. The input ceiling is 262,144 bytes total. Actual bytes: 85839.

The first input supplies 12 literal before/after rows at ticks 0, 64 and 1023, four count/error factor combinations, before mass 2, error 8.0, weight 16.0 and observation counts 8/16. After observations remain literal 9/17. All literal rows and types are copied without deriving an after-state, increment or prediction. These are standalone algebraic templates, not observed complete learner trajectories. The second input establishes only the previous 11-record stateless arithmetic check; its success is not transferred to state transitions.

## Minimum contract for the next implementation

| Boundary | Required evidence |
|---|---|
| Independent computation | A separate scalar transition implementation without primary calculation/update helper imports. Golden after values remain evaluator-only. |
| Numeric writes | Fresh output from a fixed trusted predecessor; 36 changed and 399 unchanged fields, nine route credits; reject absorbed additions. Fixture inspections must not be called actual runtime writes. |
| Identity | Separate receiving ID from shuffled source ID. Bind owner, route, key, side and predecessor independently of the candidate output. |
| Delayed outcomes | Only matured receipts; source precedes delivery. Reject same-tick/future credit and duplicate receipt. Preserve the existing fixed delay semantics rather than selecting a new delay. |
| State ownership | Separate training from diagnostic state. Diagnostic route selection cannot alter learning state; candidate, permanent and shuffled roles remain distinct. |
| Fair comparison | Same-role event-driven scalar and local engine, identical delivered results and actual update allocation. Engine labels in fixtures do not prove independent implementations. |
| Equality | Exact integer/float types, finite nonnegative numbers, no negative zero; identical binary64 hexadecimal floats and explicit no-op rejection. |
| Interpretation | Literal arithmetic, runtime write provenance, full learner equivalence, independent task usefulness and SNN causality are separate claims. |

The static module imports standard-library utilities only. It never imports the old numeric fixture constructor, old calculator, independent arithmetic kernel, learner or stream generator. It inspects frozen metadata and copies literal expectations. Thus no new state transition is applied even to a fixture in this phase. Five mutations are refused: changed literal row, missing delayed-result requirement, relaxed zero-learner budget, promotion flag and altered saved-input binding. Whole-selection hash rejection establishes immutability, not semantic verification of a future implementation. Six tests cover literal preservation, scope flags, required boundaries, mutation refusal, input roster and independent fresh output copies.

## Results and remaining gaps

The retained result explicitly marks independent transition implementation, runtime write provenance, same-role runtime scalar equality, independent task and SNN-specific causality as missing. Actual transition executions, learning, scoring, acquisition and resource measurements remain zero. All source/test seals, original gates, old replay failure, recovery/retention negatives and unknown causal explanations remain intact. No original adaptive-system pass or resource advantage follows from static contract completeness.

All six new JSON artifacts live in `data/processed/benchmark_fixtures/` under `route_scalar_transition_contract_*_v1.json`: selection, registration, execution, result, replay_execution and replay. Output bounds are 262,144 bytes per artifact and 2 MiB aggregate. Exclusive consumed markers prevent restart; retained failures cannot be retuned or retried.

🟠 [Next] Separately preregister and implement the bounded nonlearning independent scalar transition against the existing 12 literal before/after rows, including exact changed/unchanged fields and fixed identity/delay/diagnostic faults. Compare only fixture arithmetic; do not infer genuine runtime writes or authorize a learner run. Keep independent task/SNN causal and resource work pending.

## 日本語

保存済み2入力だけで、次の独立スカラー状態遷移検証の最小契約を事前登録・実装しました。既存12組の期待値を計算し直さず保持し、6テスト・5件の改変拒否・別プロセスでの静的再検証が成功、545件の既存ハッシュを維持しました。状態更新・追加学習・採点・取得・計測はゼロです。受信IDとシャッフル元、遅延結果、診断状態、同一役割比較を明確にし、実更新の由来・独立課題・SNN固有の因果根拠は未確認として保持しています。次は別途事前登録した非学習の独立状態遷移実装です。

## 简体中文

仅使用两个已保存输入，预注册并实现了下一步独立标量状态转移验证的最小契约。原样保留12组固定期望值，不重新计算；6项测试、5项改动拒绝及独立进程静态复验通过，保留545项历史哈希。状态更新、新增学习、评分、采集和测量均为零。明确区分接收ID与打乱来源、延迟结果、诊断状态及同角色比较；实际更新来源、独立任务和SNN特有因果证据仍未确立。下一步需另行预注册非学习独立状态转移实现。
