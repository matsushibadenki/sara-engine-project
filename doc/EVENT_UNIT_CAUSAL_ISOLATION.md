# Minimal Event-Computing Unit Isolation

## Research decision — 2026-09-16

[Next] The checkpoint prerequisite is satisfied under the explicit Python 3.10 runtime boundary. The current result supports the usefulness of sparse routing, local temporal state, bounded local updates and confidence routing, but it does not identify membrane potential, spike emission or refractory dynamics as the cause of the observed quality.

[Done] The immutable v1 preregistration is frozen at SHA-256 `f8c2c2d2122af7ff4691e8738a9f8b490d94e8af029195737fe04fab459feac7`. It fixes the four arms, five seeds, disjoint causal families, seven ablations, equal inputs and learning signals, resource ceilings, ordered decisions and stop rules before candidate implementation. The validator writes only a managed report and confirms that real frozen partitions remain closed.

[Done] Implemented the shared deterministic development generator and all four arms with one common bounded mistake update. Dynamic membrane/refractory state is isolated within each episode. On 250 development episodes, A and B are exactly equivalent at `0.720`; C reaches `0.912` by solving the timing and refractory families while remaining at `0.560` on branch conjunction; D reaches `1.000` and solves branch conjunction. Maximum event work is `6/6/20/22` and retained state is `2,970/2,970/19,373/26,240` bytes for A/B/C/D, within the frozen ceilings. This is development evidence only, not an accepted causal result.

[Done] Implemented all seven frozen interventions, deterministic replay digests, capacity-matched scalar state and per-seed ordered gains. The harness passes, but the development causal gate fails exactly one check: disabling refractory dynamics changes neither C predictions nor its refractory-family accuracy (`1.000`, delta `0.000`). Time shuffle drops aggregate C accuracy by `0.108`, state reset by `0.192`, spike-association shuffle by `0.144`, branch shuffle drops D branch-family accuracy by `0.400`, and outcome shuffle drops aggregate D accuracy by `0.528`. The capacity-matched scalar arm remains exactly equivalent to A despite retaining at least D's state bytes. The runner exits non-zero and held-out causal families remain unopened.

[Next] Retain v1 as a development negative result. Freeze a new, non-amending v2 protocol that separates temporal state without refractory dynamics from an otherwise matched refractory arm. Do not carry the current perfect development scores into acceptance thresholds, and do not consume the v1 held-out split.

## Factorial v2 result — 2026-09-16

[Done] Frozen v2 independently at SHA-256 `07afe1d0bd4389c1d71acecc30e360335ade36daa380579c68e8137d865a5494`, with five fresh seeds and namespace `event-unit-v2`. It compares compact events, temporal state without membrane/refractory dynamics, matched temporal state with refractory neurons, and temporal state with fixed branch structure.

[Done] The v2 harness passes but the development gate is negative. Aggregate accuracy is B `0.708`, C `0.900`, R `0.900`, D `1.000`. Temporal state adds `0.192`; time shuffle removes `0.084` and state reset removes `0.192`. Branch structure adds `0.100`, and branch shuffle removes `0.500` on the conjunction family. Refractory adds `0.000` in aggregate and is `-0.020/-0.020/0.000/-0.020/-0.020` across the five fresh seeds. Disabling refractory leaves aggregate predictions at `0.900`. Added capacity alone exactly preserves the compact-event predictions.

[Done] The v1 conclusion therefore replicates on fresh identities: local temporal pair state and fixed branch structure are useful in this synthetic scope; membrane/refractory dynamics provide no benefit and are slightly harmful in four of five seed-level comparisons. The v2 runner exits non-zero, and both v1/v2 held-out splits remain unopened.

[Next] Stop the refractory line. The smallest supported forward unit is currently a compact event plus bounded local temporal-pair state; branch structure is added only for tasks with branch-specific conjunction. Before any deeper credit mechanism, create a source-/generator-disjoint real or semi-synthetic development task that cannot be solved by an explicit pair lookup alone.

The causal comparison will use four arms:

1. **A — Scalar local learner:** no spike and no neuron dynamics.
2. **B — Compact event unit:** explicit sparse events without retained membrane or refractory state.
3. **C — Stateful spiking neuron:** retained membrane potential, spike and refractory dynamics.
4. **D — Dendritic / structural neuron:** branch-local state and structure, with structural changes disabled in the first comparison and enabled only in a separately registered follow-up.

Every arm must receive the same route identities, chronological examples, context memory, prediction head, outcome signal, update opportunities, accepted evidence and hard resource envelope. Capacity matching includes route count, learned scalar count, retained bytes, event work and tuning attempts. If an arm requires extra state, an explicit capacity-matched control receives an equal state allowance without the proposed mechanism.

The first synthetic task must independently vary four factors: order-sensitive timing, persistent state, refractory suppression and branch-specific conjunction. It must include time-shuffled, state-reset, spike-count-preserving and branch-shuffled controls. Five fixed seeds and a held-out causal-family split are required. BPI and Sepsis frozen partitions are not tuning data and must not be reopened for mechanism selection.

Primary outcomes are held-out task quality, calibration, causal-intervention sensitivity, deterministic replay, update count, event work, retained state bytes and CPU latency. A mechanism is credited only when its targeted ablation removes an advantage that survives scalar and capacity-matched controls. Exact prediction equivalence is evidence against a mechanism-specific quality claim, even if the implementation remains useful as an event representation.

Stop after the smallest successful arm. Do not implement `Local Credit Packet`, multi-layer backward messages or structural growth during this stage. Those belong to a later preregistration after the minimal forward/event unit is identified.

日本語: まず同じ routing・memory・budget・learning signal の下で、scalar、compact event、stateful spike、dendritic structure を比較します。実験runtimeは再現確認済みのPython 3.10に固定します。

简体中文：首先在完全相同的路由、记忆、预算与学习信号下比较标量、紧凑事件、有状态脉冲和树突结构。实验运行环境固定为已验证可复现的Python 3.10。
