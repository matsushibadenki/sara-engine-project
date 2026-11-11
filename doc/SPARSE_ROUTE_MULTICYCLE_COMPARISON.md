# Scalar comparison across two reactivation cycles

🟢 [Done] The preregistered synthetic run matched all 700 receipts and 6,291 local route updates between the model and independent scalar reference. It uses constant input and outcome reversals at ticks 180 and 420. Both executions start exploration at ticks 185 and 429 and replace their route pools at ticks 365 and 609. The first replacement changes two routes and the second changes nine.

The second reversal occurs during cooldown. At tick 428 the detector exceeds 0.20 and exploration is inactive. Both implementations restart exploration at cooldown expiry at tick 429. The second cooldown ends at tick 673. The comparison checks local fields, allocations, ordered credits, predictions and scheduler state throughout the run.

Saved verification checks the boundary projections and the full stream digest with model entry trapped. It rejects a missing second start and a consumed execution request. The previous one-cycle run remains pinned.

⭕️ [Pending] Independent task usefulness, external provenance, durable commit and SNN-specific causal and resource evidence.

日本語：結果を180、420 tickで反転させ、700受信と6,291局所更新が独立スカラー実装と一致した。428 tickで検出値が閾値を超えても再活性化は抑制され、429 tickのクールダウン終了時に再開した。

简体中文：在第180步和第420步反转结果，700次接收和6,291次局部更新与独立标量实现一致。第428步检测值超过阈值时仍受冷却限制，第429步冷却结束后重新激活。
