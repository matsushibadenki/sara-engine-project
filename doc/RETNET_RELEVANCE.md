# RetNet relevance to SARA

Source: Sun et al., [*Retentive Network: A Successor to Transformer for Large Language Models*](https://arxiv.org/abs/2307.08621) (arXiv:2307.08621).

## What is useful

RetNet's most relevant idea is not its large-language-model architecture. It is the separation of one sequence mechanism into equivalent parallel, recurrent, and chunkwise-recurrent execution forms. Its recurrent form updates a fixed-size decaying state at each step, so inference state and per-token work do not grow with sequence length. The paper's ablations also make fixed decay, multiple decay scales, gating, and state capacity independently testable rather than treating “memory” as one indivisible feature.

This suggests a narrow SARA hypothesis: a sparse route may maintain a few bounded, fixed-decay retention slots and use the current event to gate which local slot is read or updated. Fast/medium/slow slots can be compared under the same total state budget. Event-by-event execution is authoritative; a chunked replay form is permitted only when it is bit-exact or tolerance-equivalent and changes execution cost rather than model semantics.

## What must not be imported

RetNet itself uses dense learned query/key/value projections, outer-product state, normalization layers, feed-forward matrices, Adam/backpropagation, GPU-oriented parallel training, and billion-parameter language-model evidence. These violate or do not establish SARA's sparse, local-learning, non-matrix, CPU-first claims. The paper's reported throughput and memory gains are comparisons at its own GPU/model scale and are not evidence for SARA.

SARA must therefore not describe a scalar or sparse retention experiment as a RetNet implementation, nor use the paper to justify replacing Event Memory with recurrent state. Durable, source-aware and revisable experience remains in Event Memory; retention slots are transient local state with explicit expiry and hard occupancy bounds.

## Proposed controlled experiment

Use an independently sourced long-delay event task only after the current real-data gate work. Freeze identical routing, local learning signal, event budget and state bytes for:

1. no temporal state;
2. one fixed decay;
3. fixed multi-scale decay with equal total state;
4. content-gated fixed multi-scale decay;
5. adaptive local time constants only if the fixed controls expose a reproducible limitation.

Include time-order shuffle, decay removal, shared-decay, gate removal, reset, state-capacity, and chunk-equivalence controls. Report task quality, abstention, operations and state bytes per event, latency, peak RSS, replay equivalence, and measured energy when available. Promotion requires a preregistered quality or cost gain over the strongest fixed control on independent held-out data; otherwise retain the negative result.

## 日本語要約

RetNetから取り入れる価値があるのは、大規模言語モデル本体ではなく、同じ時系列機構を逐次・並列・チャンク逐次の等価な形式で扱う考え方、固定容量の減衰状態、複数時間尺度の分離実験です。SARAでは、各疎ルートに少数の固定減衰スロットを置き、現在イベントで局所的に読み書きを選ぶ仮説として検証します。密なQ/K/V行列、外積状態、Adam、逆伝播、GPU並列学習は採用しません。短期retentionと永続的なEvent Memoryも分離します。

## 简体中文摘要

RetNet对SARA有价值的并非其大型语言模型本身，而是同一序列机制可具有逐步、并行和分块递归的等价执行形式，以及固定容量衰减状态和多时间尺度消融方法。SARA只应测试每条稀疏路由上的少量固定衰减槽，并由当前事件局部决定读写。不得引入密集Q/K/V矩阵、外积状态、Adam、反向传播或GPU并行训练；短期retention也不能替代持久、可修订的Event Memory。
