# Dormant private-owner capture adapter candidate — 2026-11-08

🟢 [Done] Separately preregistered and implemented a concrete but **launch-disabled source candidate** for private FactorialMemory ownership and paired raw state capture. Twelve bounded static/refusal checks, eight tests and exact fresh-process replay pass; 647 historical artifact hashes are preserved. No real model is constructed or executed. This is source implementation progress, not an executed adapter, observed numeric capture or resolution of the runtime hold.

Selection SHA-256: `b45240a9c4e290492d7c0e0e93391966511c5873ad3f771d4c81649f6e34b2d3`. Source-only scope, fixed false launch gate, ownership/capture contract, check roster and budgets were saved before implementation. Candidate/harness/runner/tests/path utility and CPython 3.10.22 were registered before primary checks. Six existing source files are hash-read as evidence without importing/executing their model kernels. Only `route_pause_admission_result_v1.json` is parsed as historical input. Six create-only artifacts use `route_capture_candidate_*_v1.json`. Input limit: 128 KiB including candidate source; 256 KiB/artifact and 2 MiB aggregate.

## Concrete source path

`RouteCaptureCandidate.__init__` begins with the fixed module gate `LAUNCH_AUTHORIZED = False`; it raises before importing FactorialMemory, creating a host or model, or calling a calculator. There is no constructor permit, model argument, callback or configuration override. Capture also checks the same gate. Enabling it requires a separately reviewed/source-pinned version and execution protocol; monkeypatching Python globals is outside the cooperative trust model and is not an authorized launch path.

The unreachable model-bearing path specifies private exact FactorialMemory construction, private host/owner and one immutable Input-only envelope. The candidate does not accept an externally aliased model, an outcome, labels, true dependencies, an evaluator or a data source. It exposes immutable PhaseCapture/CapturePair records rather than the model/host capability. No observe/train/evaluate/probability/snapshot/state_digest/save/load/commit entry point is provided or called. Future permitted begin would still calculate the original prediction internally; **it has not been invoked here** and is not authorized by the static gate tests.

The candidate source takes two independently traversed typed primitive copies of `vars(model)` under the predecessor pause, performs begin only under the exclusive owner mutation interval, and takes two copies under a distinct receipt revision pause. Each pair must compare exactly with unchanged control views. A complete raw instance dictionary is recursively encoded into immutable type-tagged tuples; tuples/lists remain distinct, integers are exact strings, floats use finite hexadecimal representation, and unsupported values reject. No model method/property or outcome-derived error calculation is used by the copy path. Digests are intended to bind encoded copies; they would not authenticate origin.

The candidate yields only detached immutable input/predecessor/receipt records while the receipt pause remains active, then revalidates it. One-use state prevents reuse; stable exit closes the control host and poisoned intervals remain terminal. Source and paired raw output are bounded. There is no file writer or model persistence operation in the candidate. These behaviors are intended source semantics only: no encoding, numeric traversal, begin, host construction or capture body was run during this study.

## What the checks establish

The checker parses the candidate AST, requiring a literal false gate and a first constructor raise before lazy learner import; no top-level learner import or model/authorization argument; no selected forbidden learning/calculation/storage call; begin syntactically within owner mutation; two phase pauses; two independent read calls; frozen/slotted output records; and no direct raw model return. A constructor-refusal test confirms no additional runtime modules are imported. These are bounded syntax guards, not full call graph, alias, purity or security proofs. Name-based forbidden-call checks cannot certify arbitrary future code.

Eight tests cover current static conformance, actual constructor refusal without learner import, enabled-gate rejection, eager learner import rejection, forbidden observe/file-open calls, reuse of the first read instead of an independent second traversal, direct model export and an externally supplied model parameter. Primary and fresh-process replay match all twelve checks; registered sources and 647 historical hashes verify. Consumed primary/replay refuse before computation; the selected roadmap gate check still passes.

## Required review before any execution

⭕️ [Pending] The candidate is not runtime-ready. At least these concrete obligations remain:

- Bind a genuinely authorized runner/run epoch and a fresh unconsumed input-only source; no new source/seed is selected or read here. Hand owner labels and constructor parameters do not establish authority or unavailable future outcomes.
- Review actual full writer/alias coverage, construction failure and asynchronous interruption/cleanup semantics. Python private fields and absence of raw public return are cooperative constraints, not complete confinement.
- Verify primitive encoding against the exact 435-learning-field/150-allocation catalog, absent versus allocated-zero semantics, already-recorded route probabilities and diagnostic/scheduler completeness. Copying a raw dictionary does not yet prove that logical schema mapping or its independent readback.
- Resolve acknowledgement creation from **observed** capture digests. The earlier PauseAdmission takes an expected hand acknowledgement before capture and expects a pre-revision host; it cannot simply be relabeled as a post-capture adapter. Do not substitute hand digest labels for actual observed values or treat its fixed-plan success as runtime integration.
- Separately preregister a source-pinned bounded construction/capture protocol and stop rules before turning on any learner/host/capture call. This change grants no execution permission; unsupported fields, authority gaps or unknown paths must remain blockers.

No added hand numeric fixtures or repeated scientific scores are used to manufacture confidence. The seven missing runtime/task/SNN categories remain missing. Independent scalar producer execution, actual matched learning updates, durable nine-route commit/consumption, independent task qualification and SNN-specific causal/resource evidence remain separate holds. Original reproduction failure, negative recovery/retention/resource results, thresholds and current-state/outcome conditions and test/source seals are unchanged. Learner/calculator/model-storage/live-capture executions, task scoring, acquisition and resource measurement are zero. No production or SNN-exclusive performance claim follows.

## 日本語

実モデルを外へ渡さない前状態・受信後状態の採取アダプター候補を実装しました。ただし起動フラグは固定falseで、モデルimportより前に拒否します。12静的チェック、8テスト、別プロセス再検証が成功し、647件の既存証跡を維持しました。実モデル生成・begin・数値コピー・ホスト起動は実行していません。435学習欄・150割り当て欄との対応、実入力元の権限、実ダイジェストに基づく確認接続などは未検証で、実採取は保留です。旧失敗・負の結果・封印を維持しました。

## 简体中文

实现不导出真实模型的前状态/接收后状态采集适配器源码候选。启动标志固定false，在导入模型前拒绝。12项静态检查、8项测试及独立进程复验通过，保留647项历史证据。未构建模型、调用begin、复制数值或启动主机。与435学习字段/150分配标志的映射、真实输入源权限及基于实测摘要的确认连接仍待验证，实际采集继续暂停。旧失败、负面结果与封存保持不变。
