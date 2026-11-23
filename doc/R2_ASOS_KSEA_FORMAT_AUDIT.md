# KSEA development acquisition and format audit V1

## Preregistration

Use only the six fixed [development files](R2_ASOS_KSEA_DELAYED_CONTRACT.md) in order. One identity-encoded HTTPS GET per file, with redirects and retries disabled, a 60-second socket timeout and a 300-second total execution deadline. Require HTTP 200 and an explicit Content-Length within the registered per-month cap. Read exactly that many bytes, stop on transport or framing failure, and record size and SHA-256 per completed body. Total body ceiling 26,470,080 bytes. Keep bodies in process memory and publish aggregates only.

Use the existing timestamp parser, first-sensor normalizer and delayed outcome extractor unchanged. Count missing rows as calendar minutes minus indexed minutes, and unrecognized visibility as indexed minutes minus valid normalized minutes. Count dates with at least one valid minute and dates with at least one complete delayed outcome separately. Require at least 180 dates with eligible delayed outcomes for coverage to advance. Record class counts and exact current-key counts, repeated keys, keys with both classes and keys with rising and falling histories. Report the subset with opposite outcomes across rising/falling and the number of keys that also have at least 80 examples per direction. This count is descriptive and does not replace the existing effect threshold or qualify a predictor. Count distinct prediction dates for each class. Publish no example, key or raw record. If the structural contract fails, return inconclusive and record record-length frequencies from acquired bytes as diagnostic evidence, without relaxing the parser or returning partial coverage. Tests and one real acquisition are permitted. No training, comparison scoring, resource benchmark, July access or August access. Prior independent-task conditions and negative results remain in force.

## Result

🟢 [Done] The registered V1 acquisition stopped at the January response size gate. The admitted range was 1–4,508,640 bytes. The runner reported `inconclusive / acquisition / Body length outside registered budget`. It did not retain the exact rejected Content-Length. One GET was attempted, zero response bodies were read by the audit, and no February–June requests were made.

⭕️ [Pending] Real record omissions, unrecognized visibility, 180 eligible dates, exact-current recurrence, contrasting histories and both-class support are unmeasured. The stop gives no evidence about why the response size differs or whether the task is feasible. Synthetic verification covers separate missing/unrecognized counts, paired opposite histories and outcomes, and stopping on an oversize response before body read or subsequent requests. All 59 related tests passed.

🟢 [Done] The [metadata follow-up and transport V2 registration](R2_ASOS_KSEA_TRANSPORT_V2.md) pin the six published sizes and ETags. The V1 acquisition remains closed.

日本語　1月の応答サイズが登録範囲外だったため本文を読まずに終了した。実データでの180日・履歴・両クラスの確認は保留し、8月の封印を維持する。

简体中文　1月响应大小超出登记范围，读取正文前停止。实际数据的180天覆盖、历史差异和双类别支持仍待核实，8月继续封存。
