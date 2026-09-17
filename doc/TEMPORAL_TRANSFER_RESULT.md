# Temporal Transfer v1: Frozen Development Result

The [preregistered protocol](TEMPORAL_TRANSFER_PROTOCOL.md) has SHA-256 `10d186195b7c2fff6d441f7d8e9ff254e5a8d71198216cea060d91ca7e9789cd`. The generator/evaluator source remained `0020c04d27767efac83cd6a5513d5d8fd551ff5afef080085c46b68638c62152` and the event-unit source remained `8fbfcc8efb8f19773c789d374b670d0553ecea6acb56aa492d2f8742dd8d76e0` during the run.

Pre-execution audit: 200 training and 100 frozen-development episodes, 900 input events, zero overlapping IDs, zero exact labeled-pattern overlap, and zero nuisance-route overlap. No held-out episodes were generated. Each fixed action was prepared in one command, its intent head was recorded in the conversation before a separate execution command, and only then was the evaluator called. This record is chronological but is not an independently authenticated timestamp.

| Fixed action | Pre-evaluation intent head | Frozen accuracy / 100 | Process CPU ms | State bytes | Max event work | Prediction trace SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| B compact event | `b54c1fdf64850edf602942236464613aa486a533ce193b22a9435de18016698d` | 0.50 | 1 | 2,170 | 3 | `e36c5efd893d86292e0524ca3d6b1e9e271bb52f3cb1b25f15bf1f797614bb9f` |
| C temporal state | `a45b37cf4c23fc88bc1a76167ae60d9898f5a3788c04726a7c96ba99d4ca35cf` | 1.00 | 1 | 3,503 | 11 | `0ba6e383cc952fe4922b895ec1c2748a27e2a29343ad12b8ce00bdf90fed3635` |
| C time shuffle | `01097c33ff039802e231fa6d8b39eb8a8cb07a33c1396ca1acf66a936656ce80` | 0.50 | 2 | 3,803 | 11 | `e36c5efd893d86292e0524ca3d6b1e9e271bb52f3cb1b25f15bf1f797614bb9f` |
| C state reset | `f41be1f1a668c9b505321450a0b59a19a64a47c02ac0c490c916647085976c69` | 0.50 | 1 | 2,170 | 9 | `e36c5efd893d86292e0524ca3d6b1e9e271bb52f3cb1b25f15bf1f797614bb9f` |

For each of the five development seeds (960001, 960113, 960227, 960341, 960457), B and both controls scored 0.50 while C scored 1.00. C−B, C−time-shuffle, and C−state-reset are each +0.50, above their frozen +0.15/+0.10/+0.10 minima. The fixed state and event-work ceilings passed. A read-only rerun reproduced all four exact prediction traces and recorded scores, state sizes, and input-event counts. Development true labels were used only for scoring, never for learning updates.

Capture path: `workspace/evaluation/temporal_transfer_v1_20260917/capture.jsonl`; file SHA-256 `81ced0c752ab648926829938feb46b4b868dbe1475689f5cec533ba8d60e6487`; final chain head `c538cd09b825fdd975c04bb7890a607a5868169dd1fb002df5cf070c25db8716`. The complete projected snapshot SHA-256 is `86dad325d74243d10df6022482c14b5b934585dc43d55aa9749b2f44b571876e`. A separate read-only lineage audit matched these caller-recorded digests and all four intent heads; the projection remains **unapproved and unexported** because independent head attestation and external review are absent.

Interpretation: the bounded local temporal-pair feature transfers the deliberately shared gap rule across unseen nuisance routes and shifted absolute time origins, while the route-only arm and temporal-destruction controls do not. This is stronger split evidence than the prior 250/250-overlap capture, but the rule and decisive route pair are still shared by construction. It does not demonstrate spike-specific computation, learned latent structure, delayed multi-layer credit, a real-event gain, standard-control success, measured energy efficiency, or exploration-policy improvement. C uses more state and event work than B; 1–2 ms process-CPU readings are too coarse for an efficiency claim. Preserve this result without retuning or opening frozen BPI/Sepsis tests.

Validation: the related 47 tests pass. The repository `tests/` run passed 2,310 and failed four Phase 34 environment-fingerprint tests: their frozen manifests bind Python 3.10.20 while this host ran Python 3.10.18. Recomputing each descriptor with 3.10.20 matches all four registered digests. Do not rewrite those frozen manifests to make this host pass. An unconstrained root-level `pytest -q` also collected unrelated third-party tests under `workspace/uv-cache`; use `PYTHONPATH=. pytest -q tests` for the repository suite.

日本語: 未知の妨害routeと別の時刻原点でも、時間状態Cは100例中100例、Bと二つの対照は50例でした。入力パターン重複はゼロで、開発正解を学習には使っていません。ただし核となる時間差規則は学習時と同じであり、実イベントや深いcredit assignmentへの一般化は示しません。snapshotは未承認です。

简体中文：在未见过的干扰路由和不同绝对时间起点下，时间状态C对100个开发样本全部预测正确，而B及两项对照均为50%。跨分割的完整输入模式没有重复，开发标签未参与学习。但核心时间间隔规则保持不变，不能据此声称真实事件或深层信用分配的泛化；快照仍未获批准。
