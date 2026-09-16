# R2 BPI 2012 Confidence-Hybrid Final Gate

## Decision — 2026-09-16

[Done] The fixed confidence-routed hybrid passed every preregistered final gate on the single never-opened chronological test partition. Protocol SHA-256 is `5005b87c100290f97f2c457827e8635ebe3a22abd5dead89da1fac1c36fa3dcf`; result SHA-256 is `c84d65e619eba5a2dba986ba0a9c397e8db761f525db3a0799a5e1dca7dbd4fd`.

Across 32,694 next-activity predictions, the simultaneously updated second-order base reached 80.62% top-1, 53.60% macro-F1 and 0.2728 multiclass Brier. The hybrid reached 82.13%, 56.67% and 0.2659. These are gains of 1.52 percentage points in accuracy and 3.07 points in macro-F1, with a 0.00693 lower Brier score.

The router overrode 3,342 predictions, or 10.22%. Constant-gap macro-F1 was 56.45%, a 0.216-point loss that narrowly exceeds the registered 0.20-point timing threshold. Shuffled local outcomes reduced macro-F1 to 45.45%. The explicit-neuron and scalar arms were exactly identical.

All resource contracts passed: 1,024 routes/neurons, 5,260 stored weights, 1.59 MB state, 264 maximum counted operations, 0.187 ms p99 latency, 8.68 ms watchdog latency and 195.9 MB peak RSS.

## Adoption boundary

[Done] Adopt confidence routing as the leading isolated R2 research architecture. Keep it outside production and do not claim SNN-specific quality, because the matched scalar implementation produces the exact same predictions. The result establishes usefulness on one real irregular event log, not cross-domain generality.

[Next] Select and preregister a second independent real event log with a compatible next-event target, irregular timestamps and a redistributable source. Reproduce non-inferior accuracy/Brier, macro-F1 gain, timing ablation, shuffled-outcome dependence and bounded CPU behavior without tuning on BPI 2012.

[Later] If the independent gate passes, consolidate the generic hybrid runtime and examine whether a stateful spiking mechanism can outperform the scalar control. Measure physical energy only after a real workload and hardware procedure are fixed.

日本語: 未開封の最終testで、信頼度付きハイブリッドは正解率、macro-F1、Brierをすべて改善し、全ゲートを通過しました。BPI 2012上の主要研究方式として採用します。ただしスカラー版と完全同値であり、SNN固有の優位性や複数分野への一般化はまだ示していません。

简体中文：在从未打开的最终测试集上，置信度混合模型同时改善了准确率、macro-F1和Brier，并通过全部门槛。它被采用为BPI 2012上的主要研究架构；但由于与标量版本完全等价，目前不能宣称SNN特有优势或跨领域泛化。
