# Durable commit for a staged route receipt

🟢 [Done] SQLite publishes the post-observe state and the consumed receiving ID in one transaction. The store stages one validated capture pair per database. It binds deliveries to the stored input, acknowledgement and one-tick delay. The real model's observe method runs on a private restored copy. Publication requires nine route updates, 36 changed learning fields and a cleared pending receipt.

An immediate write transaction serializes competing deliveries. Exact duplicates return false, and conflicting duplicates raise an error. Readers fetch state and consumption in one query. The store requests SQLite FULL synchronization.

Five focused tests use the saved cold receipt. They check simultaneous duplicates, conflicting outcomes, delivery timing and process termination at three points. Termination after the state write or after both writes before commit leaves the old state and unconsumed ID. Termination after commit before the reply leaves the updated state and consumed ID, and redelivery has no effect. The tests check all 435 learning fields and 150 allocation flags after recovery.

⭕️ [Pending] The verified scope is one staged receipt per local database and abrupt process exit on the test filesystem. Multi-receipt persistence, system power loss, external authority and independent-task evidence remain open.

日本語：9経路の更新と受信ID消費をSQLiteの一括処理で保存した。強制終了後の再開、同時重複配送、競合する結果を検証した。

简体中文：在同一个 SQLite 事务内保存九条路径的更新和接收 ID 消耗，并验证进程中断恢复、并发重复投递和冲突结果。
