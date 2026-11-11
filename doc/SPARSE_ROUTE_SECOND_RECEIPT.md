# Delayed outcome and second receipt

🟢 [Done] The separately registered run captured two synthetic receipts and delivered one outcome at tick 1 after the first acknowledgement. The model allocated nine keys and changed exactly 36 learning fields. The second capture read back the updated state and produced a second acknowledgement before host closure.

The selection fixes seed 61111004, cumulative/uniform settings, two identical five-bit inputs and one literal outcome. SHA-256 `0f82102d59f942081ba3d9f784cbfffdb35aebfa26696f32aff7d47ec07cd463`. The literal is part of the registered protocol. The private runner supplies it to the bridge after acknowledgement and invokes observe after leaving the pause. This verifies ordering through the local interface. External outcome availability and authentication remain open.

The saved verifier checks each of the 435 learning fields against its expected value, including the four changes per credited route. It checks all 150 allocation flags, eight readback digests, receipt revisions 0→2 and 4→6, and the first delivery's tick, outcome and acknowledgement. Read-only verification succeeds with model entry trapped. A consumed attempt refuses before model entry. A wrong delivery tick and a wrong outcome were both rejected. The run preserves 690 earlier artifact hashes.

The scheduler selected disjoint route sets for the two receipts. The second capture contains nine populated keys in its full state and zero populated credits among its active routes. It exercises nonempty state capture and the missing-key probability branch. A future preregistration can target a repeated active route to verify an actual retained-key probability. Preserve this zero-overlap result when making that selection.

⭕️ [Pending] Retained-key probability in a live receipt, independent scalar execution, durable commit, external input and outcome provenance, independent task usefulness and SNN-specific causal and resource evidence.

## 日本語

事前登録した合成入力2件と遅延結果1件で、9経路・36学習欄の実変更と更新後状態の採取を確認した。2回の活性経路は重複せず、更新済みキーの確率読出しは保留。保存記録の検証はモデル再実行なしで完了した。

## 简体中文

用两条预注册合成输入和一次延迟结果，验证九条路径的36个学习字段变化，并采集更新后的状态。两次接收的活跃路径没有交集，已更新键的概率读取仍待验证。保存记录的复验无需重新运行模型。
