# Compact Event-Unit Backend

## Decision — 2026-09-16

[Done] The opt-in `compact_event_units` backend passed every preregistered gate on BPI and Sepsis. It compiles only the stateless route path that previously reset a full neuron before every fixed suprathreshold pulse. Stateful neuron mechanisms remain unchanged and outside this backend.

Across three independent processes per dataset and mode, BPI median CPU time was 12,865 ms for compact events versus 14,353 ms for explicit neurons, a ratio of 0.8963. Runtime state was 0.782 MB versus 1.450 MB, a ratio of 0.5390. Sepsis median CPU time was 572 ms versus 703 ms, a ratio of 0.8146; state was 0.668 MB versus 1.334 MB, a ratio of 0.5004.

Every compact run exactly reproduced the accepted frozen prediction digest and the explicit-neuron output. BPI passed the registered 0.90 CPU ratio narrowly, so the evidence supports the gate but not a broad speed claim across hosts.

[Done] Adopt compact events as the recommended CPU event backend for this stateless pulse path. Keep full explicit neurons available for equivalence tests and mechanisms with membrane, branch or refractory state. Keep scalar mode as the non-spiking matched control.

Physical energy remains unmeasured. Reduced CPU time and state do not establish joule savings.

[Next] Use the generic runtime and compact backend in future event-stream experiments instead of dataset-specific learner duplication. Any new stateful spiking mechanism must declare the state it preserves, the scalar matched control and a preregistered functional or cost advantage.

日本語: stateless pulse経路を軽量化したcompact backendは、BPIとSepsisで予測を完全に維持し、CPU時間と状態量の登録済み条件を通過しました。状態を持つニューロンの代替ではなく、現在の固定発火route専用の推奨CPU backendとして採用します。

简体中文：用于无状态脉冲路径的紧凑后端在BPI和Sepsis上完整保持预测，并通过CPU时间与状态大小门槛。它不是有状态神经元的替代品，而是当前固定触发路径的推荐CPU后端。
