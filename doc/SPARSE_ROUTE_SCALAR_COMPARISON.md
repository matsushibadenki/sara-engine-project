# Scalar replay of retained-key receipts

🟢 [Done] A separate scalar implementation rebuilds route order from the registered seed and computes its own counts, errors, weights and receipt probabilities from the fixed inputs and outcomes. The comparison checks all six saved model receipts. At each receipt all 435 learning fields, 150 allocation flags and nine ordered credits match exactly.

Run the check with `PYTHONPATH=src:. python scripts/eval/route_scalar_comparison.py`. The runner verifies pinned source and artifact hashes and the original saved result before comparison. A check with model capture trapped passed without importing the learner. An injected field mismatch was rejected.

⭕️ [Pending] The comparison covers the cumulative/uniform cell over one scheduler cycle. Broader matched execution, durable commit and independent task evaluation remain open.

日本語：固定入力と遅延結果を独立スカラー実装で再生し、保存済み6受信の学習欄、割り当てフラグ、経路と確率が完全一致した。

简体中文：独立标量实现重放固定输入和延迟结果，六次保存的接收记录在学习字段、分配标志、路径和概率上完全一致。
