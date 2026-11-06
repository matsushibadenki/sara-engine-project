# Authority-provider connection design — 2026-11-06

🟢 [Done] Separately preregistered source-only connection design: five source files, one saved capture-protocol report and ten symbol anchors. Seven tests and exact fresh-process replay pass; 617 historical artifact hashes remain unchanged. The design identifies five connection gaps and concrete insertion seams. It does not implement an authority provider or unblock actual capture.

Selection SHA-256: `fba9fb2d53f59ba9abe4f2de191dcba7428a8c8719091b16a11a28f4a45bc678`. Host ownership, revision/pause, outcome channel, readback, budgets/stops, inputs and anchors were fixed before implementation. Checker/runner/tests/path utility and CPython 3.10.22 were registered before execution. Six create-only artifacts are `route_authority_design_*_v1.json`. Input budget is 128 KiB across five sources and one saved report, ten anchors, 256 KiB/artifact and 2 MiB aggregate; live hosts/captures and permitted runtime calls are zero. Source inspection neither imports nor invokes any candidate provider or model.

## Proposed connection

A separately source-pinned, authorized runner would create a **single-owner host** that owns the model from construction. The collector must not accept a caller-supplied model/owner/revision or obtain a raw mutable model reference. The host must cover every begin, observe, scheduling, diagnostic mutation and restore operation. Locking a wrapper while raw aliases or other writers remain available does not establish exclusivity.

The trust root is that registered runner's control of the host and input/outcome channel, not the host's self-declared label. Bind a runner-issued run identity and unique host-instance epoch to source/protocol identity. Same-process cooperation is a bounded engineering trust assumption, not authentication against arbitrary Python or out-of-band label access. A new process means a new epoch; object identity, learner clock and hand digests cannot prove continuity across restart.

The host owns a monotonic mutation generation independent of receipt clock and shuffle-source address. Before any owner mutation it marks an in-progress generation; only successful completion publishes the next stable generation. Any exception poisons the host and stops processing/capture; no automatic rollback/retry or false success generation. Restore must be an explicitly authorized new epoch or separately verified path, not silent resetting of a counter. The floor is host-owned and never supplied by a collector. This proposal does not provide durable rollback protection or exactly-once storage.

A stop boundary holds the exclusive owner gate at a stable generation. No owner operations, callbacks, background writers or properties run during primitive reads. Independent readback means two independently traversed primitive copies with exact typed state equality and unchanged epoch/revision/floor before and after. The lock must span both copies and their checks, not merely each attribute access. A raw read from the publicly mutable current model outside that gate remains unsafe.

## Predecessor and receipt phases

Use a paired boundary: complete stable predecessor **before** owner begin, and an already-existing pending receipt **after** that operation, with separate phase revisions linked to the same epoch/receiving ID. Begin legitimately changes clock, pending and potentially scheduler state, so reusing one revision for both would hide a transition. In a future authorized run, only the owner may perform begin; the collector never calls begin, probability, observe or snapshot. This study does not run that sequence or reopen old executions.

The prospective input-only producer supplies events/receiver/source address before capture acknowledgement. The outcome resides in a separate unavailable channel until acknowledgement; a `Sample` with an `outcome` field is not an acceptable input-only envelope. Capture should copy pending probabilities already produced by begin, explicit allocation and complete fields from the frozen capture protocol; it must not calculate them again. Captured pre-begin and post-begin state fields must be identified separately rather than asserting that post-begin allocation/scheduler state is the predecessor. Outcome acknowledgement is a new required connection, not merely a comment placed before observe.

A consumer may receive immutable snapshots only, never model references. Caller-created epoch/revision strings are not authority. Numeric catalog encoding and actual error/learning equality remain separate validation work; this design produces no numeric snapshots. All original same-role scalar, actual-update, delay and endpoint requirements remain unchanged.

## Existing-source assessment

| Boundary | Existing seam | Required change / limitation |
|---|---|---|
| Owner construction | Factorial `run_case` constructs the model, then passes it into `train`. | A future new runner could create a private host at this seam; current mutable aliases are not exclusive ownership. |
| Mutation and stop | Budget begin writes pending/clock; observe writes tables, residuals, counters and clears pending. | No encompassing host-owned generation/pause is present at these selected boundaries. Cover all writers before calling it safe. |
| Outcome firewall | Factorial train iterates a row and reads `row.outcome` alongside begin/observe. | Existing full rows make outcomes available in the runner. A split input/outcome channel and acknowledgement are required. |
| Monotonic watermark | `AuthoritativeSequenceStore` has scoped publisher sequence, thread/file locks and persistence. | Its scope is publisher data, not model revisions. Even load enters a lock-file path; constructor may create directories. It is not invoked or reused as a read-only model authority. |
| Hand commit | Metadata commit returns a declared successor. | Does not own actual model state, enforce an exclusive pause or authenticate runtime origin. Reference only. |

These seams make the prospective design plausible **with a new host and split-input runner**, not a drop-in existing-source solution. Ten pinned excerpts record source hashes/lines and syntactic calls/attributes/stores. This is bounded source interpretation, not full transitive writer coverage, security proof or concurrent execution evidence. No conclusion is made about unrelated repository components. No persistence store, model constructor, learner, calculator or collector is executed.

## Verification and next boundary

Seven tests cover finite non-promoted output without runtime imports, repeat nonmutation, altered source/report refusal, roster/budget refusal, zero permitted runtime calls, existing load's lock path and the current runner's available outcome/mutation calls. Primary and separate-process replay agree exactly; consumed phases refuse before computation. The existing roadmap gate checker also passes; evidence documents pinned there were not changed.

🟠 [Next] Separately preregister a nonlearning single-owner lifecycle interface checking owner identity/epoch, stable and in-progress revisions, exclusive pause, alias confinement requirements and poison-on-exception transitions. No model, numeric calculator, storage backend or live attachment; this tests the newly identified control invariant, not another copy of arithmetic fixtures. A successful reference lifecycle must not be promoted to full writer coverage, external authentication or runtime capture.

⭕️ [Pending] Actual capture remains blocked until a new source-pinned host and split-input runner cover genuine model ownership, all writers, outcome order and complete readback under a separately registered execution protocol. Seven runtime/task/SNN evidence categories remain missing. Resource work, original negative results, failed reproduction, scientific conditions and test/source seals are unchanged. No SNN-exclusive benefit, live launch or production promotion follows.

Stop if runner authority, alias confinement, complete writer coverage, stable epoch/revision, outcome exclusion or independent readback cannot be established; if a helper/storage call or old consumed/sealed run is needed; or if a fixed budget is exceeded. Do not supply fictitious authority to bypass a stop. Learner/calculator/model-storage/live-capture executions, new task scores, acquisition, resource measurements and new hand model fixtures are all zero.

## 日本語

実採取の保留を解消するため、モデルを単独所有するホスト、ホスト所有の単調な世代と停止境界、入力・結果を分離する接続を設計しました。5ソース・10シンボルの静的検証、7テスト、別プロセス再検証が成功し、617件の既存証跡を維持しました。既存runnerは結果付きSampleを持ち、publisher用保存器もモデル状態版の代用にならないため、差し込むだけでは保留は解消しません。次はモデルなしで所有・版・停止・例外時隔離の制御インターフェースを検証します。学習器・計算器・モデル保存処理は実行せず、旧失敗・負の結果・封印を維持しました。

## 简体中文

为解除实际采集的阻塞，设计单一模型所有者主机、主机控制的单调代际与暂停边界，以及分离输入/结果的接口。5个源文件和10个符号的静态审查、7项测试及独立进程复验通过，保留617项历史证据。现有runner已持有带结果的Sample，publisher序列存储也不能代替模型版本权限，不能直接接入即宣称解除阻塞。下一步仅在无模型条件下验证所有权、版本、暂停和异常隔离的控制接口。不执行学习器、计算器或模型存储，保留旧失败、负面结果与封存。
