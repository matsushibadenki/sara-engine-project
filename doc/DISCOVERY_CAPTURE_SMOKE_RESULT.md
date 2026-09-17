# Discovery Capture Integration Smoke Result v1

[Done] The one-run development-only instrumentation check followed the fixed [protocol](DISCOVERY_CAPTURE_SMOKE_PROTOCOL.md). It executed only the existing `B_compact_event` arm on newly generated synthetic training/development episodes. No held-out family, BPI/Sepsis checkpoint, frozen result file, model artifact, or production path was opened or changed.

| Field | Observed value |
| --- | ---: |
| Development predictions | 10 |
| Accuracy | 0.50 |
| Generated input events | 72 |
| Retained learner state | 3,010 bytes |
| Process CPU cost | 1 ms, rounded upward |
| Pending capture intents after reload | 0 |
| Capture events after reload | 3 |

Protocol SHA-256: `af448d0a4804dad413842ad2046fa98d2bf268473ebf90e68612d9928c113d9a`. Candidate evaluator-source SHA-256: `8fbfcc8efb8f19773c789d374b670d0553ecea6acb56aa492d2f8742dd8d76e0`. Prediction-trace SHA-256: `d82f2d9e3980adb57bb5e3d96217b899cfa3bcc47e3c687bbec7d559d08bb83b`.

Capture head SHA-256: `d2c4a876f7a9e599025d7c986c73218bc4f8de2daee8c373d39cef770c559c92`. Unapproved in-memory snapshot SHA-256: `5d0e8b9c89ef08bcc7bad7c22d523305eb4a5542794cd7af4b89e4bc9bfcfb95`. The managed source log is at `workspace/evaluation/discovery_capture_smoke_v1/capture.jsonl`; its head and projection were verified by a separate readback. This document records the head but does not provide independent pre-outcome timestamping or signed attestation.

The fixed run demonstrates only that a real evaluator call occurred after a persisted intent and produced a matching captured outcome. It is a single synthetic smoke check, not a policy comparison, prospective discovery result, spike-specific gain, resource superiority, or export approval. The score of 0.50 is not interpreted as a research gate. No reviewed replay journal was created.

日本語: 先に選択を記録し、その後に評価結果を記録する経路は動作しました。正解率0.50は性能主張に使いません。

简体中文：先记录选择、再记录评估结果的路径已运行。0.50的准确率不用于性能主张。
