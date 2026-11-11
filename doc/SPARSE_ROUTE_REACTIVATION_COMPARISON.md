# Scalar comparison through reactivation and cooldown

🟢 [Done] The preregistered discounted/public-exponential run matched all 450 receipts and 4,041 actual local route updates. The outcome reverses at tick 180 under constant input. Both executions start exploration at tick 185, replace routes 5 and 7 with routes 8 and 10 at tick 365, and reach the end of the 64-tick cooldown at tick 429.

The scalar reference calculates ranking, detector state, exploration timing and cooldown from its own local statistics and the registered events. The comparison covers receipt credits, learning fields, allocation flags, predictions, pool membership and transition records at every tick. The evidence stores boundary projections and a digest of the full comparison stream. Saved verification succeeded with model entry trapped. It rejected an altered cooldown value and a consumed execution request.

⭕️ [Pending] The fixed synthetic reversal produced one reactivation cycle. Repeated cycles, external provenance, durable commit and performance on an independent task remain open.

日本語：合成結果を180 tickで反転させ、450受信と4,041更新で独立スカラー実装との一致を確認した。再活性化は185 tick、経路の入れ替えは365 tick、クールダウン終了は429 tickだった。

简体中文：在第180步反转合成结果后，450次接收和4,041次局部更新与独立标量实现一致。第185步重新激活，第365步更换路径，第429步结束冷却。
