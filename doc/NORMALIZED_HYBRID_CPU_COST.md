# Normalized-Hybrid Registered CPU Cost Gate

## Result — 2026-09-16

[Done] The registered protocol completed four unmeasured warmups and twenty measured independent processes: five repetitions for each dataset and mode in fixed-seed randomized order. Every process reproduced its accepted frozen prediction trace, and explicit-neuron/scalar outputs remained exact.

An import-path execution fault occurred before the first warmup and before any performance sample. The attempt marker was retained, the fault was recorded, and only the sibling-module import was repaired. One post-repair Sepsis scalar preflight was excluded before the registered warmups and samples.

Median BPI process CPU time was 14,671 ms for explicit neurons and 12,971 ms for scalar mode, a ratio of 1.1311. Median wall time ratio was 1.1273. Median Sepsis process CPU time was 663 ms versus 577 ms, a ratio of 1.1497; wall ratio was 1.1466. The same direction on both sources supports the narrow conclusion that explicit Python neuron activation is slower on this host.

Total-process peak RSS was dominated by parsed trace storage: 166.28 MB versus 166.31 MB on BPI and 35.83 MB versus 35.24 MB on Sepsis. This does not contradict the isolated runtime-state result, where explicit neuron objects approximately doubled model state; it shows that whole-process RSS is too coarse for BPI state attribution.

[Done] Set scalar mode as the CPU reference implementation. Retain explicit-neuron mode for exactness tests and future event-driven hardware mapping. Do not spend optimization effort on generic mechanisms until profiling attributes the measured overhead.

Energy remains unmeasured. macOS `powermetrics` is present but documents its subsystem power values as estimates; CPU time is not converted into joules. No physical-energy or cross-device claim follows.

[Next] Run scoped component profiling on the generic runtime to separate route encoding, neuron activation, score accumulation, routing, local updates and source parsing. Use the profile to decide whether a compact event-unit backend is justified. Any backend must preserve the accepted prediction digests and bounded local-learning contracts.

日本語: 独立プロセス5回ずつの登録済み計測で、明示ニューロン版はBPIで13.11%、Sepsisで14.97%多くCPU時間を使いました。全予測は完全同値です。現行CPU基準はスカラー版とし、ニューロン版は研究・ハードウェア写像用に残します。物理エネルギーは未測定です。

简体中文：在每种模式五个独立进程的注册测量中，显式神经元版本的CPU时间在BPI上增加13.11%，在Sepsis上增加14.97%，而预测完全一致。因此当前CPU参考实现采用标量模式；神经元模式保留用于研究和未来硬件映射。物理能耗尚未测量。
