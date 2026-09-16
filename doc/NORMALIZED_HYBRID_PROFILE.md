# Normalized-Hybrid Component Profile

## Result — 2026-09-16

[Done] Deterministic call profiles reproduced every accepted BPI and Sepsis prediction in explicit-neuron and scalar modes. The profiles are attribution diagnostics; profiler overhead prevents latency claims.

On BPI, `_activate` consumed 5.692 cumulative seconds with explicit neurons and 1.304 seconds in scalar mode, an excess of 4.388 seconds. Total profiled time differed by 4.162 seconds. On Sepsis, activation excess was 0.229 seconds against a 0.243-second total difference. `predict`, score/routing, `observe`, route encoding and source parsing outside activation did not show a comparable mode-specific increase.

The current route neurons are reset before every event and receive a fixed suprathreshold pulse. They therefore always fire and retain no temporal membrane state between predictions. Full `Neuron` objects implement a verified event pulse but add object mutation, branch bookkeeping and method calls without changing the accepted function.

[Done] A compact event-unit backend is justified as a narrow compiled representation of this stateless pulse path. It must remain distinct from stateful-neuron research, preserve explicit event activation, and replay both accepted traces exactly. This optimization does not establish SNN superiority.

[Next] Implement `compact_event_units` as an opt-in backend in the generic runtime. Preregister exact replay, receipt/state equivalence, bounded storage and repeated CPU improvement over full neuron objects before changing any default.

日本語: 明示ニューロンとスカラーの時間差は、両データでほぼ全てroute発火準備へ集中しました。現在のニューロンは毎回リセットされ固定パルスで必ず発火するため、状態を持たない軽量event-unit表現へコンパイルする根拠があります。

简体中文：显式神经元与标量模式的时间差在两个数据集上几乎全部集中于路径激活。当前神经元每次都会重置，并由固定的超阈值脉冲触发，因此有依据将其编译为无状态的轻量事件单元表示。
