# First-receipt integration smoke

🟢 [Done] One preregistered input reached the existing FactorialMemory in a private owner, produced a validated capture pair and generated an acknowledgement before host closure. The saved verifier checks both phase digests, all 435 learning fields, 150 allocation flags and the nine ordered receipt routes. The result occupies 17,073 bytes.

The selection fixes seed 61111003, the cumulative/uniform configuration, five input bits `(1, 0, 0, 1, 1)` and receiving/source IDs 0. The input comes from the selection literal and has no associated outcome in this protocol. The runner derives its process-local owner label from the selection hash and keeps model and capability references private. This supplies provenance for this synthetic receipt within the runner. External source authority and complete writer confinement remain pending.

Selection SHA-256 `af3211f8f2d7ed77de1663fcb494b17f9cb041c8b31d142326c1fa1d28a00c6d`. The selection and source registration precede the exclusive execution reservation. The separate runner uses the unchanged model, host, layout reader and acknowledgement bridge. The capture candidates retain their frozen source and launch flags. The result's call-budget record describes the source-registered schedule. The saved state checks provide the direct evidence for zero learning updates.

The schedule constructs one model and calls begin once. It traverses the live primitive state twice under each of the two pauses. Revision 0 binds the predecessor and revision 2 binds the receipt. The acknowledgement binds routes `21, 6, 12, 31, 37, 13, 23, 17, 41` in recorded order. Both copies agree in each phase. The raw states differ only in `clock` and `pending`. All allocation flags are false and the update counter stays at zero. The host ends closed. The run covers cold state and one begin call, which includes the model's existing prediction path. Outcome delivery, observe, scoring and model persistence are outside the registered schedule.

The runner exposes `register`, `execute` and `verify`. Verification reads saved artifacts and performs layout and binding checks without importing the learner or executing capture. A check with capture replaced by a failure trap passed verification and rejected a second execution before model entry. In-memory corrupt receipt digest and partial-route mutations were rejected. Source and evidence verification covers 686 earlier artifact hashes. Evidence files use `route_first_receipt_*_v1.json` in the managed benchmark fixtures directory.

⭕️ [Pending] Nonempty model state, outcome ordering and delivery, matched scalar execution, durable commit, an independently qualified task and SNN-specific benefit remain separate verification work. The cold receipt result grants no evidence on those outcomes. Earlier failures, negative results and sealed splits retain their recorded status.

## 日本語

実モデルを使う接続確認として、事前登録した合成入力1件を初期状態で受信しました。前状態と受信状態の二重読出しが一致し、9経路の確認を生成してホストを閉じました。変化は `clock` と `pending` のみで、学習更新は0件でした。保存記録の検証はモデルを再実行せず、再実行要求はモデルの生成前に拒否します。実データ由来、独立課題、SNN固有の効果は保留です。

## 简体中文

对一个预注册的合成输入完成了实际模型的首次接收验证。前状态和接收状态各自的两次读取一致，生成九条路径的确认后关闭主机。仅 `clock` 和 `pending` 发生变化，学习更新为零。保存记录可在不重新运行模型的情况下验证，重复执行请求会在创建模型前被拒绝。真实来源、独立任务及 SNN 特有优势仍待验证。
