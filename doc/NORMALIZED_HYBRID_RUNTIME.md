# Generic Bounded Normalized-Hybrid Runtime

## Milestone — 2026-09-16

[Done] `BoundedNormalizedHybrid` consolidates the accepted real-event mechanism into one dataset-independent component. It owns bounded second-order context counts, sparse local class weights, normalized or absolute confidence routing, one identity-bound pending receipt, local post-prediction updates and optional explicit route neurons. It uses no matrix operations, GPU or global backpropagation.

[Done] State export rejects pending predictions and records schema, configuration, labels, routes, weights, contexts and sequence. Loading revalidates every bound. `save()` accepts a plain filename only and writes atomically under `models/normalized_hybrid/` with flush and `fsync` before replacement. English, Japanese and Simplified Chinese status text is available.

[Done] Generic-runtime replay exactly matches both immutable accepted frozen prediction traces. BPI replay covers 249,113 online events and 32,694 frozen predictions; Sepsis covers 14,164 events and 2,035 frozen predictions. Explicit-neuron and scalar modes also match each other exactly.

The single-process CPU samples are diagnostic, not a physical-energy or stable performance gate. On this run, BPI used 48.30 microseconds/event and 1.45 MB state with explicit neurons versus 40.29 microseconds/event and 0.75 MB for scalar execution. Sepsis used 34.59 versus 27.19 microseconds/event and 1.33 versus 0.63 MB. The current Python neuron representation therefore adds cost without changing predictions.

[Done] Retain explicit-neuron mode for research equivalence and event-driven hardware mapping. Prefer scalar mode for the current CPU reference runtime. Do not claim SNN efficiency from this implementation.

[Next] Freeze a repeated cost protocol with warmup, randomized mode order, multiple processes and physical energy evidence when a supported meter is available. Separate neuron object overhead, route encoding and shared learning work. Only port measured hotspots after the protocol identifies them.

日本語: 汎用ランタイムはBPIとSepsisの採用済み予測列を完全再現しました。現行Pythonではニューロン版はスカラー版より遅く、状態も大きいため、CPU実行ではスカラー版を基準とします。これはSNNの可能性を否定するものではありませんが、現在の実装から効率優位を主張することはできません。

简体中文：通用运行时完整复现了BPI和Sepsis已采用的预测序列。当前Python神经元版本比标量版本更慢且状态更大，因此CPU参考运行时采用标量模式。该结果不否定SNN的可能性，但不能据此宣称当前实现具有效率优势。
