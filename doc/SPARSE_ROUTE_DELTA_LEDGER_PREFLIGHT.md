# Nonlearning delta-ledger adapter preflight — 2026-10-29

🟢 [Done] Separately preregistered **read-only symbolic delta accounting** implements the next interface boundary without applying learner updates. Full 435-slot views enforce exactly **36 changed / 399 unchanged symbolic stamps** per receipt. **144 boundary fixtures**, **31/31 rejected faults**, **221 pure tests**, exact fresh-process replay and **509 preserved historical hashes** pass. Actual learning updates, model runs, outcomes, predictions and new snapshots remain zero; old failures, negatives, original gates and sealed periods are preserved.

## Registration and finite scope

The [selection](../data/processed/benchmark_fixtures/route_delta_ledger_preflight_selection_v1.json) predates implementation and pins the [receipt/state/scalar interface](SPARSE_ROUTE_INTERFACE_PREFLIGHT.md). Sources, tests, dependencies and Python 3.10.22 are registered before the new symbolic execute/replay phases. No old consumed phase or scientific run is restarted.

The [adapter](../src/sara_engine/evaluation/route_delta_ledger_preflight.py) consumes externally supplied **immutable before/after views** and updates only its own bookkeeping pointer after full validation. It contains no learner callback, update equation, sampled event generator, outcome payload, probability query or prediction scorer. It does not load model states. The views are synthetic symbolic fixtures, not new captures of any model.

The finite fixture scope is seed1201 / independent / changed metadata, four count/error factor combinations × two engine descriptors × three roles × ticks **0,63,64,511,512,1023** = **144 transactions**. Each boundary starts from a fresh symbolic fixture; it is not a continuation, reset or replay of a trained model. The public-time/change boundaries exercise identities and transaction shape only; no weight or behavior effect is measured.

## Complete field catalog

| Symbolic slot family | Count | Meaning in a future real adapter |
|---|---:|---|
| Two class-mass slots per route-key slot | 300 | Both class counters, including the uncredited side |
| Observation/error/weight slots per route | 135 | Three fields for each of 45 routes |
| Total | **435** | Entire fixed-capacity core view |

The 45-route catalog retains all 150 key slots, both class sides and all route-level fields. Every `CoreView` requires the complete canonical address sequence and a correctly typed opaque digest for every slot. Missing, extra, duplicate, reordered, mutable or coerced addresses are rejected, preventing a partial view from hiding inactive-field changes.

Key-slot indices and `mass_left` / `mass_right` are **schema placeholders**, not generated input bits, actual observed keys or sampled labels. No semantic class/value inference follows from them. A future numeric adapter must independently bind these slots to actual key/outcome provenance.

## Pinned credit and exact delta coverage

At `open`, pin the parent's exact nine active route IDs in their original order, plus one symbolic key slot and one mass side per route. Receipt, case and shuffled source must agree with the parent metadata plan. The pending credit tuple is immutable and cannot change when the delayed delivery arrives.

At `accept`, require:

1. The pending receipt, credit tuple and complete case identity match.
2. Delivery owns that receipt ID; shuffled source remains provenance only; delivery is declared training credit at own tick+1.
3. The before view equals the adapter's current committed view; the after view has the same case owner and core channel.
4. Both diagnostic handles equal the current diagnostic handle during a core transaction.
5. The exact changed-address set is the observed-side mass plus observation, error and weight for each of the nine active routes: **36 stamps**. All **399 other stamps**, including inactive routes/keys and the opposite mass side, must remain identical.

Only after all checks succeed does the adapter replace one immutable `LedgerState` pointer, clear pending and advance the metadata tick. No learning value is calculated or applied. Receipts explicitly identify `scope=symbolic_delta_fixture`, `actual_learning_updates=0` and `runtime_numeric_delta_tested=false`.

A changed digest witnesses only a **symbolic identity difference**. It does not prove a positive numeric increment, correct magnitude, finite probability/error calculation or genuine non-no-op numeric write. The parent's future 9,216 real route updates / 36,864 real field changes remain requirements, not achievements of this preflight. The 144 accepted fixtures contain **5,184 symbolic stamp changes**, which are never counted as actual learning.

## Atomic rejection and channel ownership

Malformed proposals or any delta mismatch leave the adapter's state object identical (`state_after is state_before`). The caller's before/after views are immutable, and the adapter never mutates a learner. A valid proposal can still be accepted after an invalid one is rejected, showing that rejection has not partially consumed pending credit.

`DiagnosticProposal` is a separate path. It requires no pending receipt, a current before view, an unchanged complete core after view and the current diagnostic predecessor. The accepted replacement retains the exact original core object. A diagnostic proposal cannot alter even one core slot. Conversely, a core proposal cannot silently replace diagnostics. Direct frozen-core mutation and mutable list aliases are rejected.

These are API/adapter ownership guarantees over immutable fixtures, not a security sandbox against privileged Python mutation and not proof about future unimplemented learner memory. Declared training-purpose/source metadata is not payload authentication. Real runtime provenance, ownership and numerical delta observations remain pending.

## Registered fault injection

All **31 registered faults** are rejected with unchanged adapter state:

- Source used as receipt ID, wrong source, same-tick or late delivery, evaluation-purpose delivery, foreign case, missing pending and duplicate pending receipt.
- Stale before view, foreign after-view owner and diagnostic channel passed as core.
- Changed pinned key or mass side, reordered credits and missing active-route fields.
- No-op mass/observation/error/weight stamps; opposite-class, inactive-route or inactive-key writes.
- Extra, missing or duplicate field addresses, mutable field alias, diagnostic drift during core acceptance and duplicate acceptance.
- Numeric payload disguised as a stamp, diagnostic proposal changing core and frozen-core mutation.

The complete ordered roster, rejection exception types and state-identity checks are retained in the report. Test cases also reject boolean address coercion, verify all 435 addresses and both namespace paths, and confirm fixture/protocol immutability and finite-scope refusal.

## Verification, preservation and claims

Eleven new tests plus 210 prior pure checks give **221 passing tests**. No learner test, generator, training runner, score evaluation or actual scalar implementation is invoked. Independent fresh processes produce the same canonical symbolic report. Consumed phases refuse restart before computation; read-only verification hashes old evidence and rechecks only symbolic fixtures/faults.

The result is **4,306 bytes**, under the preregistered 256 KiB result limit; the complete new artifact set remains below 2 MiB. **509 historical artifact hashes** and source/runtime pins verify. The old tuple/list reproduction failure, unknown discarded-run equality, 256-tick recovery failures, independent-task/resource negatives and seals remain unchanged. Original thresholds and matched-control endpoint restrictions are not modified.

The adapter retains local scalar field accounting and the independent event-driven scalar comparator design. It uses no GPU, matrix or global-backprop dependency, but supplies no SNN-exclusive evidence, energy/resource claim, production promotion or original adaptive-system success. Same-role numerical scalar equality is still untested. No additional learning is authorized by this result.

- [selection](../data/processed/benchmark_fixtures/route_delta_ledger_preflight_selection_v1.json): `cc77d029b8d7a902b1ccba8ce1537b647309dcbe479927d3225968f21ea074c0`
- [registration](../data/processed/benchmark_fixtures/route_delta_ledger_preflight_registration_v1.json): `3aa7412da7a6814d053ff4a8e79ef5b3d94faa2ece359805bf48d525f824e6f1`
- [result](../data/processed/benchmark_fixtures/route_delta_ledger_preflight_result_v1.json): `61c8406f5824c4dd703ad8411338a760ea76dc220818268641a889e389ceb6e7`
- [replay](../data/processed/benchmark_fixtures/route_delta_ledger_preflight_replay_v1.json): `9f4ed8bfaef50b0160adb3b3ec2d01a9171ad35a8849ae8495dcdbd730bc72e8`

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/route_delta_ledger_preflight.py verify
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m unittest discover -s tests -p test_route_delta_ledger_preflight.py
```

🟠 [Next] Preregister a separately versioned **nonlearning numeric snapshot-validator contract** using bounded, hand-authored before/after fixtures only. Specify numeric types, finite values, signs, magnitudes and no-op detection in addition to the now-tested address/provenance/ownership checks. No learner update, task stream, sealed data, numerical engine replay or adaptive acceptance should be implied by fixture validation.

⭕️ [Pending] Real per-route updates, observed key/outcome provenance, independent scalar numerical equality, independent-task/SNN-specific causal evidence and resource promotion remain on hold for a separately authorized version. No old restart or condition adjustment.

## 日本語

全435欄の不変な前後状態表を照合し、9経路の36欄だけが変わり399欄が不変であることを検査する非学習の差分台帳アダプターを事前登録・実装した。受信時にキー・クラス欄を固定し、配送時のID・元添字・遅延・所有者の不一致、非対象欄の変更、変更漏れ、診断状態の混入など31故障をすべて拒否した。144検証用データ、221テスト、別プロセス再現一致、509件の過去ハッシュを確認した。5,184件は識別用指紋の変化で、実際の学習更新ではない。数値差分・実ラベル・スカラー数値一致は未検証とし、旧失敗・負の結果・封印と元の判定基準を維持する。

## 简体中文

预注册并实现了不执行学习的差分台账适配器，检查完整435个不可变字段，仅允许9条路径的36个符号标记变化，其余399个保持不变。在接收时固定键槽和类别槽，拒绝ID、来源、延迟、所有者、未选字段变化、遗漏变化和诊断状态混入等31种故障。144组符号样例、221项纯测试、独立进程精确重放及509个历史哈希验证通过。5,184次标记变化不属于实际学习更新。数值差分、真实标签与标量数值等价仍未验证，保留旧失败、负面结果、原始门槛和封存。
