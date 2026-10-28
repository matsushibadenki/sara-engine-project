# Receipt/state/scalar interface fault preflight — 2026-10-28

🟢 [Done] Separately preregistered **metadata-only interfaces and finite fault injection**. Immutable receipt/source identities, separate core/diagnostic handles and same-role scalar comparison packets are implemented without a learner, outcome payload, generated task input or prediction scorer. **31/31 registered faults are rejected**, **15,360 receipt/delivery metadata pairs** and **240 symbolic comparison pairs** pass, and fresh-process replay is exact. **210 pure tests** pass; 503 prior hashes and all old failures, negative results and seals remain intact.

## Registration and scope

The [selection](../data/processed/benchmark_fixtures/route_interface_preflight_selection_v1.json) was written before implementation, pinning the [matched-control contract](SPARSE_ROUTE_FIXED_CONTROL_CONTRACT.md) and its preserved history. Code/tests and dependencies are frozen before new metadata-only execute/replay phases. No earlier attempt is resumed or modified. The prior scientific conditions, fixed allocation, genuine four-write requirement, event-driven scalar design, gate/channel restrictions and offline shuffle remain unchanged.

This implements a usable **metadata/provenance boundary**, not a numerical learner or actual model-state adapter. Ten new pure unit tests combine with 200 prior pure checks for 210 passing tests. No v4 development/learner tests are invoked in this verification.

## Interfaces

| API | Implemented boundary |
|---|---|
| `CaseKey` | Typed seed/task/condition/count factor/error factor/engine/role identity |
| `Receipt` | Case, own receipt ID, shuffled source index, exact immutable active/readout ID tuples |
| `Delivery` | Case, own receipt ID, source index, own delivery time and declared training/evaluation purpose; **no outcome-value field** |
| `ReceiptGate` | One pending metadata receipt; validate every identity/address/time before changing gate state |
| `CoreHandle` | Frozen opaque digest, with no numeric learning state or learning update method |
| `DiagnosticHandle` | Separate frozen opaque digest and immutable candidate-pool tuple |
| `StateBundle` / `replace_diagnostic` | Return a new diagnostic wrapper while retaining the exact same core object and digest |
| `ComparisonPacket` / `compare_same_role` | Compare exact receipt/core/primary/shadow fingerprints across the two registered engines within the same role and conditions |

The [implementation](../src/sara_engine/evaluation/route_interface_preflight.py) imports only the earlier **pure static** address/contract helpers, not generator, learner, prediction scorer or numeric evaluation modules. The bounded source-map cache holds five immutable index tuples; callers cannot mutate these through list aliases. Code comments and logs are English.

## Receipt ID is not label provenance

`make_receipt` derives addresses from the unchanged public tick and source metadata from the registered role. Candidate/permanent source index equals the tick; shuffled source equals the fixed phasewise `pi(t)`. Both receipt and delivery carry the complete case identity, including engine and role, preventing foreign envelopes from being accepted merely because their integers happen to coincide.

`ReceiptGate.open` validates the expected case, own sequential receipt ID, shuffle source and exact active/readout ordering before storing one pending item. `deliver` requires its own pending receipt ID, matching source, same case, declared training purpose and **time=receipt_id+1**, then clears pending and advances the metadata step. A source index may refer to a later label, but it never becomes the receipt identity or dispatch time. Evaluation-purpose envelopes are refused at this boundary.

Fifteen complete metadata traces cover five seeds × three roles × 1,024 receipts, including the final receipt1023/delivery1024 boundary. Total: **15,360 receipt/delivery pairs**; each trace ends at 1,024 steps with no pending receipt. No inputs, bit vectors or outcomes are generated or credited. These traces exercise metadata sequencing, not actual learning.

The purpose flag checks **declared provenance only**. Without payloads or an actual engine, it cannot certify that a future caller has not mislabeled a true/evaluation value as a training value. Future adapters must verify payload provenance and receipt/source bindings, not rely on a caller-supplied string alone. The offline shuffle remains a negative control, not live causal label availability.

## Core and diagnostics are separate

All handle and packet dataclasses are frozen and use slots. Mutable active/readout/pool aliases, malformed digests and core handles passed to the diagnostic replacement API are rejected. Diagnostic replacement preserves `new_bundle.core is old_bundle.core`; the original diagnostic wrapper is unchanged. Normal field assignment to a frozen core is rejected.

No diagnostic callback receives a writable numeric core or learning function in this interface. This is API/ownership preflight using opaque fixture handles, not a security sandbox against privileged Python mutation, and not a proof that an unimplemented numerical engine will respect the boundary. Actual runtime ownership, update-ledger checks and core-state isolation stay pending.

## Same-role scalar comparison

Comparison requires exactly one `local_sparse_reference` and one `independent_event_driven_scalar` case key. Seed, task, condition, history, error weighting **and role** must match. Primary, shadow, core and receipt-ledger fingerprints must each match exactly. There is no numerical tolerance, coercion, ignored field or cross-role comparison. A true-label candidate cannot be compared to a shuffled-label model as if they were equivalent implementations.

The 240 fixture pairs span five seeds × two tasks × two conditions × two count rules × two error rules × three roles. Each fixture deliberately supplies matching **symbolic opaque digests** to the two engine descriptors. This verifies the interface's matching/rejection logic; it runs neither engine and supplies no numerical evidence about either. Returned receipts explicitly say `scope=metadata_fixture`, `runtime_scalar_equality_tested=false` and `original_adaptive_pass=false`. Attempts to label these packets as runtime numerical equality are rejected.

Actual same-role independent scalar equality, genuine 9,216 route updates / 36,864 changed fields per prospective model, true frozen evaluation and original adaptive acceptance remain untested. No new scientific success or threshold failure is inferred from fixture equality.

## Registered fault injection

The immutable protocol lists 31 ordered fault cases. The retained report records each rejection, exception type and unchanged metadata/core handle checks. The fault set covers:

- Shuffled source used as receipt ID, wrong source, same-tick/late outcome, boolean ID, duplicate pending receipt and delivery without pending.
- Evaluation-purpose metadata entering the training boundary, wrong/reordered active addresses, wrong primary readout and foreign role/seed/engine receipt envelopes.
- Mutable active-address/pool aliases, core passed as diagnostic state and frozen-core mutation.
- Cross-role, seed, task, stationary/changed, count-factor and error-factor comparisons, plus comparing an engine to itself.
- Core, primary, shadow and receipt-ledger fingerprint mismatches, malformed digest and attempted runtime equality claim.

All **31 faults are rejected before changing the gate state or core fixture handle**. Unit tests additionally check foreign delivery role/engine envelopes, input immutability, exact registered coverage and rejection of changed fault/budget contracts. The after-rejection equality checks are on the metadata gate/core fixtures, not on uninstantiated learner weights.

## Verification, budgets and preserved negatives

The [runner](../scripts/eval/route_interface_preflight.py) verifies **503 historical artifact hashes**, source pins, parent binding and Python 3.10.22. The result is **53,139 bytes**, below the 256 KiB per-artifact cap; all six new artifacts remain below the 2 MiB total cap. Two fresh metadata executions produce exact canonical result equality. Consumed phases refuse restart before computation; read-only verification recomputes metadata and faults only.

Current counters remain zero: learner executions, actual learning updates, outcome payload reads, generated samples, prediction scoring, new snapshots and resource measurements. No old successful replay is fabricated. Old tuple/list reproduction failure, unknown discarded-run equality, 256-tick recovery failures and independent-task/resource negatives remain unchanged; no sealed period is opened.

The design retains the strong event-driven nonspiking scalar comparator. It uses no matrix/GPU/global-backprop dependency, but demonstrates no SNN-specific benefit. Original adaptive-system gate acceptance cannot be obtained from fixed-readout, diagnostic selection or symbolic comparison packets. No promotion or further learning is authorized by these checks.

- [selection](../data/processed/benchmark_fixtures/route_interface_preflight_selection_v1.json): `aefb9ce748c621f9fe46d9fe48562316d4223218331027560fe0c3bebc9af630`
- [registration](../data/processed/benchmark_fixtures/route_interface_preflight_registration_v1.json): `d27a26dfe5956273d3340633b53fe5048a21eabdd1e32b32c0ccd2f9498857bd`
- [result](../data/processed/benchmark_fixtures/route_interface_preflight_result_v1.json): `9a4bf9e79c6fe1d52ce23fd311ada863246a235caa86139a8a31b2accf3f706e`
- [replay](../data/processed/benchmark_fixtures/route_interface_preflight_replay_v1.json): `99896b42879c4d74ebd3661b9df33077ec794c5d2b1ed8fea19e01489df44749`

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/route_interface_preflight.py verify
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m unittest discover -s tests -p test_route_interface_preflight.py
```

🟠 [Next] Preregister a finite nonlearning delta-ledger adapter contract: receipt/source/key identity, active/inactive field accounting, rejection atomicity and channel ownership, with symbolic fault fixtures only. Keep actual learner updates and numerical scalar equality pending a separately authorized version rather than reclassifying this interface preflight as runtime validation.

⭕️ [Pending] Numerical learners, payload provenance validation, real per-route update checks, independent scalar equality, fresh independent-task/SNN-specific causal evidence and resource/energy promotion remain on hold. No automatic old restart, additional learning or condition tuning.

## 日本語

受信IDとシャッフル元、学習状態と診断状態、同一役割の独立スカラー比較を分離する不変メタデータAPIを事前登録・実装した。15,360組の受信・配送、240組の象徴的な指紋比較と31故障の拒否を検証し、210テスト・別プロセスの再現一致・503件の過去ハッシュを確認した。診断差し替えは同じ学習状態ハンドルを保持し、不正入力の拒否時には保留状態を変えない。実ラベル・学習器・数値状態は扱っておらず、スカラー数値一致や実更新の証明ではない。旧失敗・負の結果・封印と元の条件を維持し、適応系の合格やSNN固有優位性は主張しない。

## 简体中文

预注册并实现了不可变元数据API，分离接收ID与打乱来源、学习状态与诊断状态，以及同一角色的独立标量比较。验证15,360组接收/配送、240组符号指纹比较，31种故障全部拒绝，210项纯测试和独立进程精确重放通过，503个历史哈希保持一致。诊断替换保留原学习状态句柄，拒绝非法输入时不修改待处理元数据。未处理真实标签、学习器或数值状态，因此不证明实际更新或标量数值等价。保留旧失败、负面结果、原条件和封存，不宣称自适应系统通过或SNN独有优势。
