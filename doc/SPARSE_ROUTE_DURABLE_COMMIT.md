# Durable commit for a staged route receipt

🟢 [Done] SQLite publishes the post-observe state and the consumed receiving ID in one transaction. The store stages one validated capture pair per database. It binds deliveries to the stored input, acknowledgement and one-tick delay. The real model's observe method runs on a private restored copy. Publication requires nine route updates, 36 changed learning fields and a cleared pending receipt.

An immediate write transaction serializes competing deliveries. Exact duplicates return false, and conflicting duplicates raise an error. Readers fetch state and consumption in one query. The store requests SQLite FULL synchronization.

Five focused tests use the saved cold receipt. They check simultaneous duplicates, conflicting outcomes, delivery timing and process termination at three points. Termination after the state write or after both writes before commit leaves the old state and unconsumed ID. Termination after commit before the reply leaves the updated state and consumed ID, and redelivery has no effect. The tests check all 435 learning fields and 150 allocation flags after recovery.

⭕️ [Pending] The verified scope is one staged receipt per local database and abrupt process exit on the test filesystem. Multi-receipt persistence, system power loss, external authority and independent-task evidence remain open.

日本語：9経路の更新と受信ID消費をSQLiteの一括処理で保存した。強制終了後の再開、同時重複配送、競合する結果を検証した。

简体中文：在同一个 SQLite 事务内保存九条路径的更新和接收 ID 消耗，并验证进程中断恢复、并发重复投递和冲突结果。

## Sequential receipts in one database

🟢 [Done] The database retains the consumed-ID ledger across receipts. A separate checkpoint exposes the latest committed revision and state while the next receipt is pending. The first staged predecessor provides the initial checkpoint. Each subsequent stage requires the next receiving ID, the same owner and epoch, the exact previous committed state and the next predecessor revision. The receipt and outcome each advance the revision by two.

`stage_next` restores a private model from the checkpoint, runs begin and stages the validated receipt. The write transaction checks continuity again before publishing. Applying an outcome commits the state, checkpoint and consumed ID together. An exact duplicate of any committed receipt returns false, including while a newer receipt is pending. Conflicting duplicates, future deliveries and replacement of pending receipts are rejected. Legacy committed databases derive the checkpoint from their stored state and acknowledgement.

Eight focused tests pass. The sequential recovery test commits the first receipt, terminates a child before the second commit, then terminates a retry after commit and before reply. A reopened store suppresses redelivery and stages a third receipt from the second committed digest. The final state contains 27 route updates. The tests also cover skipped revisions and continuation from the original schema.

⭕️ [Pending] External delivery authority, cross-database coordination and system power loss remain outside the verified local process interruption scope.

日本語：同一DBの3受信で、状態版の連続性、過去の重複、順序違いの配送、2件目のコミット前後の中断と3件目への再開を確認した。

简体中文：在同一数据库中验证三个连续接收，覆盖版本连续性、历史重复、乱序投递、第二次提交前后的中断以及第三次接收的恢复。
