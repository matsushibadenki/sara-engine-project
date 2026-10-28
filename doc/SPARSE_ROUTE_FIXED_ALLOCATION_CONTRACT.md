# Fixed-allocation local probability comparison contract — 2026-10-28

🟢 [Done] Separately preregistered **design and static verification only**. The contract fixes both learning addresses and the primary readout, isolating local evidence effects from adaptive allocation feedback. An independent event-driven scalar engine is mandatory. Static proofs and **189 pure tests** pass with exact fresh-process static replay. **No learner, task generator, prediction scoring or new capture runs.** Actual learning updates remain zero. Original failures, negative results, thresholds and sealed periods remain unchanged.

## Why fix two paths

The [saved factorial audit](SPARSE_ROUTE_FACTORIAL_AUDIT.md) found that nine of ten history-factor pairs already differ in route allocation before the task changes. Holding only the total update count fixed does not hold each route's exposure fixed. The proposed intervention therefore fixes the **same active route IDs at each receipt** and **the same primary readout route IDs** across count factors, error factors and both implementations. Future runs must also match event/key/label receipt digests before claiming equal exposure.

Scores and a shadow copy of the original selector may change. Their predictions feed only their own detector/lifecycle; they must never select learning addresses, change main readout IDs, alter priors or update other routes. This removes selection-to-training feedback while retaining a separate diagnostic of ranking/readout effects. It is not an attempt to replay or silently replace the old adaptive schedule.

The fixed schedule cannot demonstrate adaptive scheduling benefit or original adaptive-system recovery. The fixed primary readout deliberately uses no true dependency IDs; it may be poor at the original task. No adaptive accuracy baseline or full component success can be inferred from it. Equal-address probability contrasts, shadow ranks and primary fixed-readout behavior are separate endpoints.

## Preregistered finite design

The immutable [selection/contract](../data/processed/benchmark_fixtures/route_fixed_allocation_contract_selection_v1.json) predates implementation. Source/test registration precedes static execution and a separate-process static replay. The only data structures processed are this contract, static route addresses and receipt identities; earlier artifacts are read for preservation hashes, not to extract or score samples.

The prospective design retains the existing two synthetic tasks, five seeds (1201/1213/1229/1237/1249), stationary/changed conditions, 1,024 receipts, change at 512, warmup 180, binary inputs/outputs, 16-route-observation threshold, width cost 0.005, trigger threshold 0.20, detector rate 1/16, exploration 180, cooldown 64 and fixed 64-receipt recovery blocks. No existing learner source is edited, imported or executed. Reusing these known seeds would remain a mechanism study, not independent confirmation.

| Design dimension | Fixed contract |
|---|---|
| Count history | Cumulative or public-time discounted |
| Error weighting | Uniform or public-time exponential |
| Engines | Local sparse reference and independently implemented event-driven scalar |
| Minimal future scope | 4 factor cells × 2 engines × 20 cases = 160 model runs per phase |
| Prospective phases | New primary + fresh-process replication; neither is authorized or started |
| Current work | Static address/receipt proofs only; zero learning |

The minimal future scope has **no matched permanent or shuffled-outcome arms**. It cannot pass their original gates. Resolving these controls requires a separately registered scope before any training; the 160-run count is a conditional ceiling for the minimal design, not a budget for silently adding controls or an execution approval.

## Fixed allocation and shared inputs

Use the unchanged catalog of 45 width-one/two routes in lexicographic input-combination and output order. Each output has an ascending list `group` of 15 IDs. At public receipt tick `t`:

```text
active routes per output = group[(3*t+j) % 15], j=0,1,2
fixed primary readout per output = group[(3*t) % 15]
```

The primary readout abstains until that route has at least 16 observations, and for an unseen dictionary key. The address function accepts **only the integer receipt tick**: no labels, events, scores, dependency truth, change hint, case identity or donor adaptive trace. Shared/overlapping input routes remain in the full catalog; no input is dropped to improve a contrast.

Exhaustive static enumeration verifies nine distinct active routes per tick, three per output. Across 1,024 receipts, **36 routes receive 205 visits and nine receive 204**, totaling **9,216 planned route updates**. Before tick 80 every route has exactly 16 planned observations. This is a static receipt-address count, not a trained observation table. The schedule and fixed-readout sequence have a retained deterministic hash.

## Local rules and actual-write contract

Keep `w(t)=2**floor(t/64)`. Cumulative evidence increments only the observed class by one, using prior one. Discounted evidence increments only that class by `w(t)` and uses prior `w(query_tick)` in `(C1+prior)/(C0+C1+2*prior)`. Thus the effective prior remains one observation per class. The clock and weight are public, not outcome-dependent. Frozen query time stays 1023; no decay driven by frozen row order. Absent keys abstain.

Independently, update error numerator and normalizer with weight one or `w(t)`. Route score stays `E/W + 0.005*key_width`, default error 0.25 before support. With the same route/key/label receipts and count rule, changing only error weighting **must leave count tables and route probabilities identical**. It can change scores/shadow rankings, not main exposure. This is a required future runtime invariant, not a measured equality here.

Each active route must genuinely change exactly four learning fields: observed-class mass, route observation count, error numerator and error normalizer. Therefore each prospective cell/case requires **9,216 actual route updates / 36,864 changed fields**. Future verification must inspect before/after values, independently reconstruct all active field deltas and verify inactive fields and key allocations unchanged, plus per-route observation, weighted-mass and weight conservation. Dummy writes, no-op equalization, bulk decay or resets are prohibited. Scheduler/detector metadata writes and arithmetic operation counts are not matched, so this does not establish equal total processing or energy.

A finite static binary64 bound supports the no-op requirement: maximum 205 observations per route, weight at most 32,768, and maximum class mass/weight/error-sum bound **6,717,440**, below exact integer range `2**53`. Effective evidence never exceeds 205, so the smoothed class probability and its complement are at least `1/207`. A positive squared-error increment is at least `1/207**2` (about 0.00002334), far above twice the largest-sum ULP (about 0.000000001863). Hence the registered finite equations admit genuine positive error/weight changes. This mathematical bound is not a substitute for observing actual updates; runtime confirmation remains pending.

## Delayed results and event-driven scalar control

Receipt `t` must predict from prior state and retain its own active IDs and probabilities before seeing its label. Exactly one pending receipt is allowed. Only its own outcome at `t+1` may credit those routes; duplicate receipts, missing receipt, same-tick/late result, wrong identity, boolean timestamp or nonbinary result must be rejected before mutation. Clear pending after update. Frozen evaluation requires no pending receipt and no further learning or clock advance.

The implemented static trace validator checks **receipt metadata only**, not binary outcome values, prediction or learning. A 1,024-receipt trace has 2,048 event dispatches, maximum one pending item, zero idle dispatches and zero expiries. Wrong identities/timing/order are tested without generating task inputs or outcomes. Binary payload rejection and real-buffer behavior are explicit future implementation obligations; they are not reported as tested runtime features.

The independent scalar control must use its own scalar dictionaries and receipt/pending/outcome state machine, the same factors/equations/tie ordering/packets and identical initial state. It dispatches only receipt and due-outcome events; no polling or idle clock scan is permitted. The shared observation packet must be equally available, including shared inputs. Record accounting separately for observer, schedule, main lookups/readout, four local writes, due-result dispatch, pending-buffer allocation/release and shadow selection/detector. Waiting performs zero dispatches without an event. Expiry is zero in this complete one-tick stream; invalid late results receive no synthetic learning credit. Future exact same-cell equivalence must cover receipt digests, probabilities, scores, primary readout, shadow metadata and frozen outputs.

This is a strong event-driven **nonspiking** comparator, not a dense or polling ANN straw control. The contract defines no SNN-exclusive mechanism or SNN arm. Sparse local scalar operations are compatible with SARA's no-global-backprop/no-matrix/no-GPU constraints, but compatibility alone proves no spike-specific advantage. Any later SNN claim requires a separate causal mechanism and fair independent comparison.

## Original gates retained, applicability restricted

| Original gate | Unchanged requirement |
|---|---|
| Changed contrast | Initial newly required dependency absent |
| Tail exact accuracy | ≥0.90 |
| Gain over permanent | ≥0.10 |
| Recovery | First two consecutive post-change 64-receipt blocks each ≥0.90; ≤256 ticks after change |
| Dependency retention | Every true dependency retained |
| Outcome alignment | Tail gain over shuffled outcomes ≥0.20 |
| Frozen exact accuracy | ≥0.90 |
| Stationary false starts | ≤1 |
| Stationary noninferiority | Permanent tail minus candidate tail ≤0.02 |

Missing recovery remains `None`, never zero. Fixed-readout performance and shadow retention/recovery do not replace original adaptive-system gates. Without matched controls, permanent/shuffled gates are **untested, not passed**. No gates are evaluated in this static phase. The prior 320-tick recovery failures, old tuple/list reproduction failure and unknown discarded old-run numerical equality remain unchanged. No source expansion, real-data acquisition, sealed-period reads, independent-task success, resource measurement or production promotion occurs.

## Static verification and retained proof

Eleven new pure tests cover all 1,024 address rows and readout balance, all-eight-cell address identity, fixed-period budgets, absence of learner imports/dispatch, unchanged gates/conditions, metadata timing/identity errors, scalar-control presence, support timing, input immutability and prohibitions on oracle inputs. Together with 178 prior pure checks, **189 tests pass**. Static execute/replay phases are exclusively consumed, immutable and refuse restart before computation; verification recomputes only addresses, arithmetic bounds and receipt metadata.

**491 prior artifact hashes** verify. Static execution has zero actual learning-field changes, generated task samples, new predictions/snapshots, acquisitions or resource measurements. Per-artifact cap is 256 KiB and total cap 2 MiB. Planned future counts are explicitly distinct from actual current execution counters.

- [selection](../data/processed/benchmark_fixtures/route_fixed_allocation_contract_selection_v1.json): `84b926ef42392565c836c180eaadba15694935cc3451effeb87b48ea12caa896`
- [registration](../data/processed/benchmark_fixtures/route_fixed_allocation_contract_registration_v1.json): `0056265a54d720fc8fc58a0a35541259086afeaf56b188e984554dde2ed24fb4`
- [result](../data/processed/benchmark_fixtures/route_fixed_allocation_contract_result_v1.json): `98c03d8c18605dbccd519810e4255b8a48f5c2fa6a6bf0c76678e48f538f004c`
- [replay](../data/processed/benchmark_fixtures/route_fixed_allocation_contract_replay_v1.json): `ceeab6994401b5c1b06da288dc3b3ad9f098f62d0b83792488cc397dc1371d1a`

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/route_fixed_allocation_contract.py verify
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m unittest discover -s tests -p test_route_fixed_allocation_contract.py
```

🟠 [Next] Resolve a finite, nonexecuting matched-control and endpoint-applicability specification for this fixed-allocation design: keep all original gates, identify which can be evaluated under fixed exposure, and specify permanent/shuffled controls without turning shadow metrics into adaptive-task acceptance. No learner execution is authorized by this contract.

⭕️ [Pending] Future two-engine learning implementation/execution requires a new version and registration after matched-control scope is complete. Independent-task/SNN-specific causal evidence and resource/energy promotion remain on hold. No automatic old restart or condition adjustment.

## 日本語

更新配分と主読み出しを全条件で固定し、順位・プール・検出器は学習配分に戻さない独立診断経路に分離する契約を事前登録した。4因子条件×局所参照・独立イベント駆動スカラー実装の8セルを定義し、1,024受信で9,216経路更新・36,864フィールド実変更となる計画を静的に検証した。遅延1 tick・保留1件・不正順序拒否と、有限期間の非ゼロ更新の数値上限を確認したが、実際の学習・スカラー数値一致は実行していない。元の256 tick回復基準などは維持し、永久固定・結果シャッフル対照が未定義の最小設計では合格扱いしない。189テストと静的再現の完全一致、491件の過去ハッシュを確認。旧失敗・負の結果・封印を保持し、SNN固有の優位性は主張しない。

## 简体中文

预注册了固定更新地址和主要读取地址的比较契约，将排名、候选池和检测器限制在不反馈训练分配的独立诊断路径。定义4种因子条件×局部参考和独立事件驱动标量实现，共8个单元。静态验证1,024次接收对应9,216次计划路径更新和36,864次计划字段变化，并验证一tick延迟、单一待处理项、非法时序拒绝及有限数值范围内的非零增量。没有实际学习或标量数值对照运行。保留256 tick等原始门槛；缺少永久固定和打乱结果对照时不得判为通过。189项纯测试及静态精确重放通过，491个历史哈希保持一致。保留旧失败、负面结果和封存，不主张SNN独有优势。
