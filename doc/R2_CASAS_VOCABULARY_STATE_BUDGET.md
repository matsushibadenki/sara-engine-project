# Known-combination and unknown-event state ceilings — 2026-10-17

🟢 [Done] Preregistered count-address analysis certifies the pooled representation's logical route count under a fixed, explicitly identified event vocabulary. It distinguishes new combinations of known events from genuinely unknown event types, and leaves unrestricted future-event feasibility unresolved. No predictive result is reclassified, learner changed or source opened.

## Fixed inputs and vocabulary definition

Protocol `4d6f343ed05e3c02e5cb3dccf9c1548f45cf004f9e82b01497cd134a9d63302f` and code hashes were registered before the new count-address inspection. It pins 302 earlier JSON/gzip artifacts. Inputs are the same HH104 80-day fitted `pool_evidence` and `joint_evidence` checkpoints under `models/casas_pooling/`; each retains 52,659 historical changed writes and the same exact current counts. Acquisition and new learning remain **zero** for both arms.

An event type is the full `(sensor identifier, binary message)` pair. It is not merely a sensor or room name. The known vocabulary is the union of events appearing in the stored current keys and in the joint model's two stored lag-event positions. The pooled booleans cannot recover those lag identities. Only retained keys are inspected; events that occurred elsewhere in the raw stream but never entered those keys are not claimed to be known.

The counts reveal **19 current event types, 20 stored-history types, and a union of 20**, including **one history-only type**. Using only the 19 current types would understate the known vocabulary. The deterministic event inventory and its hash are retained separately for replay. No raw sensor rows or outcome labels beyond the already stored fit counts are read.

Current conditions remain exactly the raw sensor/message, weekday, hour and 15-minute clock bin. The strict disjoint evidence threshold `log(80)`, previous class/support/quality conditions, update budget and representations remain unchanged. There is no unknown-event rejection, remapping, vocabulary admission rule or current-state coarsening.

## Preregistered route envelopes

Let V be the number of event types allowed by a stated scenario. There are `C=7×24×4×V=672V` possible current addresses. Every model includes one global address and the current-address dictionary.

- Pool: at most four detail addresses per current (sensor-repeat boolean × fast-gap boolean). Total route upper bound is **1+C+4C = 1+3360V**.
- Joint: two lag events give V² choices; two four-bin gaps give 16 choices. Total route upper bound is **1+C+16CV² = 1+672V+10752V³**.

These are conservative syntactic envelopes. They allow all combinations, including histories that may not actually be jointly reachable. Four pooled branches is still a valid upper bound when a one-sensor vocabulary cannot generate every boolean combination. Each count route stores two outcome counters; these are logical dictionary-route counts, not bytes, RAM, energy or total runtime bookkeeping. Integer counter size and other memory costs are not certified by a route-count bound.

The analysis distinguishes three worlds:

1. **Exact-address recurrence:** only already stored keys recur. The ceiling is the existing number of routes.
2. **Known-event recombination:** current events and both lag events come from the fixed union vocabulary, while clock and history combinations may be new. Apply the full envelope at V=20.
3. **Unknown-event growth:** U genuinely new event types give the envelope at V+U. If U is unrestricted over an unlimited horizon, no finite global route ceiling follows. With the existing finite acquisition ceiling B=263,295 additional qualified outcomes, the generic bound `S+2B` still applies: each outcome can introduce at most one current and one detail key, while global already exists.

Where two upper bounds apply, their minimum is reported. These assumptions are not mixed with the earlier fixed-rate recurrence forecast to claim predictive success: allowing new combinations changes the observed-frequency assumption in that forecast.

## Retained measurements

| Scenario | Pooled routes | Joint routes | Capacity certificate |
| --- | ---: | ---: | --- |
| Existing exact addresses | 9,219 | 15,322 | Both within 100,000, assuming exact recurrence |
| All combinations over 20 known event types | **67,201** | **86,029,441** | Pool certified; joint not certified |
| Known events, also limited by finite acquisition | 67,201 | 541,912 | Pool certified; joint not certified |
| One genuinely new event type (21 total) | 70,561 | 99,588,385 | Pool certified conditionally; joint not certified |
| Unrestricted unknown types, finite acquisition B | 535,809 | 541,912 | Neither guaranteed within capacity |
| Unrestricted types and unlimited horizon | No finite bound | No finite bound | Unresolved |

The largest total vocabulary cardinality whose complete envelope is <=100,000 is **29 for pool** and **2 for joint**. For pool, 29 types give 97,441 routes; 30 give 100,801. Thus, relative to the retained 20 types, the pooled envelope can certify at most **nine additional event types**. These are sensor/message pairs, not nine new sensors. This is a mathematical certificate frontier, **not a rule to discard the tenth unknown type**. No input policy is implemented. Joint already exceeds the envelope certificate at its known vocabulary, so it has no certified additional-type allowance under this envelope.

Exceeding a conservative upper bound does not prove that an actual stream will overflow: the stream may occupy only a small subset of that envelope. Conversely, the fitted vocabulary does not establish that future events will stay inside it. The previous unrestricted-event budget limitation is therefore preserved. The new result is a more precise **conditional capacity property**, not evidence of temporal usefulness, an SNN advantage or feasibility of acquiring enough useful observations.

## Verification, artifacts and seals

135 related tests pass. Checks include unchanged fitted count/current/global/write conservation, matching current hashes, recovery of history-only event types, exhaustive one-clock-cell toy route enumeration against the original address functions, monotone integer capacity frontiers, invalid saved gap rejection and a witness where an unknown lag changes a pooled branch without changing the current address. State inputs remain unmodified. Fresh-process result and event-inventory replay are exact; all 302 earlier JSON/gzip artifacts remain hash-identical.

Protocol, registration, consumed execution marker, inventory, result and replay receipt are under `data/processed/benchmark_fixtures/casas_vocabulary_budget_*_v1.json`. No model is created or modified. Replay requires only tracked fit checkpoints:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/casas_vocabulary_budget.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_casas_vocabulary_budget.py
```

Exclusive artifact creation prevents consumed runs from silently replacing their evidence. The audit reads **zero raw rows and zero confirmation/development/test predictions**. The existing 4,249,759-record seal and older household seals remain in force without rescanning data. All prior zero-activation and failed quality results remain negative.

🟠 [Next] Before any new comparative learner, preregister a bounded training-only vocabulary/arrival preflight on a fresh identity. Measure qualified-outcome yield, repeated-context support and raw-event/address growth within the fixed observation/write/state ceilings; preserve the exact current and evidence conditions and all independent-period seals. State assumptions must be checked explicitly rather than enforced by dropping unknown events. A newly observed vocabulary is not an all-future guarantee, and no independent validation or runtime promotion follows without the original positive-component/quality prerequisites.

🔴 [Later] Preflight acquisition requires its own bounded preregistration. Comparative learning, independent validation and runtime/factor integration remain subject to the original usefulness prerequisites. None is executed in this count-only study.

日本語: 既存カウントの現在状態19種類と過去位置のイベントを合わせると、既知イベントは20種類でした。この20種類の新しい時刻・履歴の組合せまで含めた上限は、集約67,201セル、非集約86,029,441セルです。集約は100,000セル以内を条件付きで保証できますが、未知イベントが無制限に増える場合は未確定です。29種類までの上限保証は入力拒否規則ではありません。取得・学習は0回、閾値・現在状態・既存の負の結果と封印は維持しています。

简体中文：保存计数中的19类当前事件与历史位置事件合并后，共有20类已知事件。允许这些事件产生新的时钟和历史组合时，聚合上界为67,201单元，非聚合为86,029,441。聚合在固定词表假设下可保证低于100,000，但无限未知事件增长仍未确定。最多29类的容量证书不是输入拒绝规则。实际采集与学习均为0，证据阈值、当前状态、既有负结果和全部封存保持不变。
