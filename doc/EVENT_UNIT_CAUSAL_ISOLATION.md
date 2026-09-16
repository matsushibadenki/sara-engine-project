# Minimal Event-Computing Unit Isolation

## Research decision — 2026-09-16

[Next] The checkpoint prerequisite is satisfied under the explicit Python 3.10 runtime boundary. The current result supports the usefulness of sparse routing, local temporal state, bounded local updates and confidence routing, but it does not identify membrane potential, spike emission or refractory dynamics as the cause of the observed quality.

[Done] The immutable v1 preregistration is frozen at SHA-256 `f8c2c2d2122af7ff4691e8738a9f8b490d94e8af029195737fe04fab459feac7`. It fixes the four arms, five seeds, disjoint causal families, seven ablations, equal inputs and learning signals, resource ceilings, ordered decisions and stop rules before candidate implementation. The validator writes only a managed report and confirms that real frozen partitions remain closed.

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
