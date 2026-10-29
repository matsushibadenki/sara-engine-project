# Two-phase pre-outcome witness seal — 2026-10-29

🟢 [Done] Separately preregistered **nonlearning two-phase interface** excludes current class/error credit from the pre-outcome packet, seals receipt/key/allocation/probability and binds class only through its own delayed-result entry point. **72 completed hand fixtures /648 route bindings**, **11 pre-only records**, **26/26 rejected faults**, **253 pure tests**, exact fresh-process replay and **527 preserved historical hashes** pass. No learner, sampled result, performance scoring, model capture or sealed access occurs.

## Why the phase boundary matters

The [previous binding preflight](SPARSE_ROUTE_ERROR_BINDING_PREFLIGHT.md) verified equations on complete hand-authored records. It did not establish that probability/key/allocation were recorded before a class became known. Earlier symbolic credit tuples included a mass side at fixture-open time; they remain narrower schema examples, not a causal online interface or real result evidence.

This separately versioned API has **no class or error field in its pre packet**. It validates and fixes a before-state lookup independently of any output-side vector, and accepts class credit only at the delayed-result phase. Old protocols, source files, reports and negative outcomes are preserved unchanged. No existing scientific condition, gate, seed or learning rule is adjusted.

## Registration and interfaces

The immutable [selection](../data/processed/benchmark_fixtures/route_two_phase_preflight_selection_v1.json) was written before implementation and includes the exact 26-fault roster and finite fixture scope. Source/test/runtime registration precedes new execute/replay phases. No consumed historical phase is restarted.

| Interface | Contents and restrictions |
|---|---|
| `PreLookup` | Route ID, symbolic key slot, allocation flag, prior-state probability. No current class or error increment. |
| `PreOutcome` | Receipt identity, immutable numeric before view, nine `PreLookup` records. No result, after view, output vector or class-bearing credit tuple. |
| `SealCapability` | Immutable deterministic fingerprint plus process-local object identity. Copied/foreign objects cannot finish the pending seal. |
| `DelayedClass` | Own delivery metadata and common three-output class-side vector. |
| `TwoPhaseGate` | One pending sealed pre packet; validate the complete delayed phase before one bookkeeping-state replacement. |

The [implementation](../src/sara_engine/evaluation/route_two_phase_preflight.py) contains no learner callback or numeric update function. The before view's historical class counters are legitimate prior state; they are distinct from the current result class, which is excluded from the pre interface. The fixture case descriptor is audit context, not proof that an actual learner has been denied external/private information.

## Phase 1: seal without class credit

`seal(pre)` first validates the exact parent receipt/source/fixed addresses, complete immutable before view, route order and all key/allocation/probability records. For seen keys it recomputes the original Laplace lookup from the before masses and public receipt-tick prior. For absent keys it requires both zero masses and `probability=None`; allocated zero-count keys instead retain probability0.5. No output-side vector or error witness is needed or accepted to perform this lookup check.

The seal fingerprint includes full case/engine identity, own receipt ID, shuffled source, fixed active/readout IDs, typed before-state signature and lookup records. Float values retain exact hexadecimal identity. The pending packet and token are frozen. There is no method or finish keyword to replace pre probabilities, keys or allocation flags.

This is a **process-local capability**, not a cryptographic timestamp, signature, remote transport protocol or serialized capability. An equal copied digest is insufficient; the exact pending capability object must be presented. Digests in retained reports are deterministic evidence of fixture contents, not authenticated observation time. Privileged Python object mutation is outside the API guarantee; ordinary immutable-field assignment is rejected and the fingerprint is rechecked at finish.

## Phase 2: bind only the own delayed class

`finish(capability, result, proposal, witness)` requires a pending seal and its exact capability. The delayed result must have the same case/receipt/source, declared training purpose and **time=own receipt+1**. Shuffled source remains provenance and cannot become a receipt ID or clock. Evaluation, early, late and foreign records are rejected before changing state.

The numeric proposal's predecessor must exactly equal the sealed before view, and its route/key credits must match the sealed lookups. Class sides are then matched to the three-output vector from the delayed-result record. Only here are class-bearing `LookupCredit` records constructed from the sealed probabilities and post-phase error witnesses. The unchanged probability/error binding and numeric/structural validators check the original arithmetic, exact deltas, inactive slots and diagnostic isolation.

Successful validation consumes the seal once and advances only the interface's metadata tick. Invalid requests preserve the exact phase-state object, leaving the valid seal available for a correct finish. Duplicate finishes and resealing the completed receipt are refused. No learner state is changed.

## Finite hand fixtures and tests

The twelve fixed numeric templates from the parent × two engine descriptors × three roles yield **72 finished fixtures /648 route bindings**. Their class vectors are hand-authored offline; they are supplied only through the post-phase API during the check. They are not sampled task outcomes or numerical-engine outputs.

Eleven additional hand records exercise **pre-phase only** with asymmetric masses, both public-prior boundaries and absent versus allocated-zero keys. Their manifests also contain class/error values for prior arithmetic tests, but the new pre-phase constructor deliberately does not forward those fields. Each pre-only fixture is independently sealed and discarded without a result; it is not a live pending queue or a measured expiry experiment.

All 26 faults reject with unchanged phase state:

- Premature class/error keywords, post-credit objects passed as prelookups and a result before any seal.
- Duplicate seal, copied/foreign/tampered token, completed-receipt reseal and duplicate finish.
- Replacement prelookup/allocation keywords, frozen probability mutation, changed before view or changed key credit.
- Wrong receipt, source used as receipt ID, wrong source, same-tick/late/evaluation/foreign result.
- Inconsistent class vector, stale error witness, malformed result vector and wrong numeric after value.

Eleven new unit tests cover these faults, class-free field schemas, exact seal consumption, absent/zero prelookup and successful retry **after a rejected request without consuming its seal**. These are interface requests within one fixture, not retries of consumed research phases. With 242 previous pure checks, **253 tests pass** without importing or invoking a learner, generator or prediction scorer.

## Evidence, preservation and limitations

The result is **4,251 bytes**, below the fixed 256 KiB per-artifact ceiling; all new artifacts remain below 2 MiB. The [runner](../scripts/eval/route_two_phase_preflight.py) verifies 527 prior artifact hashes, parent binding, all sources and Python 3.10.22. Fresh processes reproduce the canonical result exactly. Research execute/replay phases are immutable and refuse restart before recomputation; read-only verification recomputes hand-fixture checks only.

The phase machine proves **API call order and fixture immutability**, not that a caller lacked the result beforehand. A fixture author may know the offline class template. Before-state values, allocation flags, declared training purpose and delivery clock remain trusted metadata, not observed or authenticated event facts. A real implementation still needs independently verified observation/result provenance and runtime state ownership. The object-capability scheme does not provide security against a hostile process with unrestricted Python access.

Current actual-learning, sampled-task, outcome-read, performance-score, new-snapshot and resource counters remain zero. No actual scalar engine runs and no same-role numerical equality is established. The old reproduction failure, unknown discarded-run equality, 256-tick recovery failures, independent-task/resource negatives and sealed periods remain intact. Fixed-readout/shadow/fixture results do not establish original adaptive-system acceptance or any SNN-specific benefit, energy advantage or production promotion.

- [selection](../data/processed/benchmark_fixtures/route_two_phase_preflight_selection_v1.json): `5c8a562798daf7b2c31a18dae5593cedeec8b4b2dadd6c7ce390225f540b64bf`
- [registration](../data/processed/benchmark_fixtures/route_two_phase_preflight_registration_v1.json): `7cd5e6213e11edde8c2e33659f3f3abc0e725daa363c34529b72a0eaff87d0c4`
- [result](../data/processed/benchmark_fixtures/route_two_phase_preflight_result_v1.json): `ae3f2ed59277ba738bfd59d2159b91eb8d7bb8f4b6443452550c75ef6d47edfd`
- [replay](../data/processed/benchmark_fixtures/route_two_phase_preflight_replay_v1.json): `cf79974744c08d5fd8677b8a1436979c2976b2df460d05427676cda50226cba9`

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/route_two_phase_preflight.py verify
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m unittest discover -s tests -p test_route_two_phase_preflight.py
```

🟠 [Next] Preregister a finite **saved-proof readiness/gap audit** before further scaffolding or training. Distinguish tested fixture/API invariants from missing runtime provenance, independent scalar computation and independent-task/SNN causal evidence; identify the smallest justified next check and keep execution/resource holds when evidence is insufficient. No new samples, learning, scoring, acquisition or resource measurement.

⭕️ [Pending] Authentic runtime observation/result timing and allocation provenance, actual numerical learner updates, independent scalar equality, independent-task/SNN-specific causal evidence and resource promotion remain on hold for separately authorized work. No automatic training or old restart.

## 日本語

受信・更新前状態・キー／割当／確率だけを事前に固定し、クラス信用は自身の1 tick遅延結果で初めて結び付ける二段階APIを事前登録・実装した。72件・648経路、事前記録のみの11例、26故障の拒否、253テスト、別プロセス再現一致、527件の過去ハッシュを確認した。複製・別APIの識別子、確率やキーの差し替え、早着・遅延・別受信の結果、二重完了を拒否し、拒否時の状態を保持する。手書き例に対する呼び出し順と不変性の検証であり、実環境で結果が未知だったことや観測時刻の真正性は認証しない。追加学習・条件調整なしで、旧失敗・負の結果・封印を維持する。

## 简体中文

预注册并实现两阶段API：先固定接收、更新前状态、键/分配/概率，之后只通过自身延迟一tick的结果绑定类别信用。72组/648条路径、11个仅预阶段记录、26种故障拒绝、253项纯测试与独立进程精确重放通过，527个历史哈希保持一致。拒绝复制或外来标识、概率/键替换、提前或迟到及错误接收的结果和重复完成，拒绝时保持状态不变。这只验证人工样例的API顺序和不可变性，不认证真实环境中的结果未知性或观测时刻。没有新增学习或条件调整，保留旧失败、负面结果和封存。
