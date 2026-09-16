# R2 Sepsis Source-Normalized Final Gate

## Decision — 2026-09-16

[Done] The fixed source-normalized hybrid passed every preregistered gate on the single never-opened Sepsis test partition. Protocol SHA-256 is `114ccfdacf9747f537d33ff03c85dc56b78f8c7a016b50bd848cb875d1b35707`; result SHA-256 is `472c55315bbaf6c54c914b5425426e43fb34b093096086b632916a99085bf4a1`.

Across 2,035 next-activity predictions, the simultaneously updated second-order base reached 55.97% top-1, 35.43% macro-F1 and 0.5643 Brier. The normalized hybrid reached 56.81%, 40.31% and 0.5625. These are gains of 0.84 percentage points in accuracy and 4.88 points in macro-F1 with a 0.00182 lower Brier score.

The candidate overrode 200 predictions, or 9.83%. Constant-gap macro-F1 was 40.03%, a registered 0.278-point timing loss. Shuffled local outcomes reached 34.81% macro-F1, below the 35.43% base. Explicit-neuron and scalar predictions remained exactly identical.

All resource contracts passed: 1,021 routes/neurons, 3,997 stored weights, 1.46 MB state, 177 maximum counted operations, 0.041 ms p99 latency, 0.207 ms watchdog latency and 37.6 MB peak RSS.

## Research decision

[Done] Record limited cross-source R2 replication. The sparse local hybrid improved accuracy, macro-F1 and Brier on both the BPI loan-process log and the independent Sepsis hospital log while retaining timing and shuffled-outcome controls. The Sepsis final gate used a prospectively amended relative shuffled-uplift condition after the training-only absolute-gap gate was found incompatible with a 10% override budget; this limitation remains part of the evidence.

[Done] Do not claim SNN-specific predictive superiority. The explicit-neuron and scalar traces are identical on both sources. The demonstrated mechanism is a bounded, event-driven, local-learning hybrid that can be implemented with spiking units, not evidence that spikes improve quality.

[Next] Consolidate the duplicated evaluation implementation into a generic bounded normalized-hybrid runtime with explicit receipts, managed serialization and replay equivalence. Benchmark explicit-neuron and scalar modes on both frozen streams for work, memory and measured CPU/energy before any production or SNN-efficiency claim.

[Later] Test a stateful spiking-only hypothesis only if it has a preregistered functional distinction from the scalar control. A mathematical scalar simulation of the same dynamics remains the required matched control.

日本語: 未開封Sepsis testで正解率、macro-F1、Brierをすべて改善し、BPIとは異なる病院ログでも限定的に再現しました。ただしスパイク版とスカラー版は完全同値です。局所・疎・イベント駆動方式の有効性は示しましたが、SNN固有の予測優位性はまだ示していません。

简体中文：在未打开的Sepsis测试集上，准确率、macro-F1和Brier均得到改善，并在不同于BPI的医院日志上实现有限复现。但脉冲版本与标量版本完全等价，因此目前证明的是局部、稀疏、事件驱动混合机制的有效性，而不是SNN特有的预测优势。
