# Scalar comparison at the first pruning boundary

🟢 [Done] The preregistered synthetic run matched all 182 receipts and 1,629 actual route updates in the discounted/public-exponential cell. The scalar reference calculates the per-output ranking from its own error and weight accumulators, including the key-width cost and route-ID tie break. At tick 180 both executions narrow to the same nine routes, and their tick-181 state matches after the first update to that pool.

The run uses seed 61112002 and constant registered inputs and outcomes. Comparison covers every learning field, allocation, receipt credit, prediction, pool and detector value. It stores projections at ticks 179, 180 and 181 and the digest of all comparisons. Saved verification succeeds with model entry trapped. A reordered pool and repeated execution are rejected.

⭕️ [Pending] The detector stays below the reactivation threshold in this run. Reactivation, cooldown and subsequent pool replacement need separate coverage. Independent task usefulness, SNN causality, durable commit and resource comparisons remain open.

日本語：枝刈り前後を含む182受信で、独立スカラー実装と完全一致した。180 tickで選ばれた9経路と、その更新後の状態も一致した。再活性化は別の検証範囲とする。

简体中文：跨越首次剪枝的182次接收与独立标量实现完全一致。第180步选出的九条路径及其更新后的状态均一致。重新激活仍待验证。
