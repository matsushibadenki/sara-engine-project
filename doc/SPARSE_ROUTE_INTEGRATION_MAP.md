# Source-only host integration map — 2026-11-06

🟢 [Done] Separately preregistered five source files, one saved host report, bounded lexical scan rules, eight manually reviewed dispatch edges, required changes and stop conditions before implementation. All 62 module functions/direct class methods are inventoried. Six tests and exact fresh-process replay pass; 629 historical artifact hashes are preserved. No learner, numerical calculator, storage adapter, host lifecycle or live attachment is executed.

Selection SHA-256: `47dbb8582b92496f1a7ca4893e6bcb23c25e77d65c97636fa0885200b597c10c`. Scanner/runner/tests/path utility and CPython 3.10.22 were registered before primary execution. The five pinned sources are sparse_factor_routing, sparse_route_budget, sparse_route_reactivation, route_memory_factorial_v4 and route_owner_host. Only `route_owner_host_result_v1.json` is parsed as historical input. Six create-only artifacts use `route_integration_map_*_v1.json`. Limits: 128 KiB total input, 16 alias-propagation passes per function, 256 KiB/artifact and 2 MiB aggregate. Result size is 101,029 bytes.

## Update and access map

| Existing boundary | Required host-side change before attachment |
|---|---|
| Local/Budget/Reactivation/Factorial constructors | Create privately inside an authorized owner. Do not accept an already-shared mutable model or export tables, mutable diagnostic state or model handles. Constructor does not itself establish external authority. |
| Reactivation begin → Budget begin | Gate the whole inherited operation, including narrowing, prediction, scheduling, probabilities, pending assignment and clock change. Record pre-begin and post-begin phases separately. The collector calls neither operation. |
| Budget begin → Reactivation narrow | Gate pool/pruned/initial-pool changes, exploration and cooldown metadata, detector reset, starts/commits list updates and nested shortlist/score work. A gate around only count updates misses these writes. |
| Budget begin → inherited Local predict, Budget select, Factorial probability, Reactivation schedule | Owner-only internal dispatch. Bound dynamic targets by source identity; do not permit an unreviewed subclass/callback. Read-looking names do not prove transitive purity. These calculators are not collector entry points. |
| Factorial observe override | Gate count `setdefault`, local count alias writes, observations/errors/weights, detector, update/step counters and pending clear as one owner interval. The override does not execute Reactivation observe as well. Preserve actual-update and no-op conditions. |
| Base Local/Budget/Reactivation observe | Inventory and quarantine alternative direct entry points, even though the selected Factorial dispatch overrides them. No external caller may bypass the owner using a base method or raw alias. |
| Factorial train/run_case, Reactivation train/run_case, budget train | Replace raw model argument handoffs with private-owner commands. Current synchronous local aliases are still aliases; no lock inside the model protects an external reader/writer. |
| snapshot, state_digest, evaluate/evaluate_cell | Existing diagnostics/evaluation access model internals, recompute scores/probabilities or pass/clone models. Keep separate from raw capture. Frozen evaluator identity and labels must not enter a receipt collector. |
| deepcopy(model) / copied table and residual lists | Conservative scan records provenance through copies. A copy is not automatically a shared mutable alias, but also not the same authoritative owner/epoch. Review snapshot independence, new owner origin and restore/restart behavior separately. |
| OwnerHost lifecycle methods | Current host covers control metadata only; it owns no model and cannot prove the above writers are routed through it. No host method is executed by this map. |

Manual dispatch links are recorded as an explicit source-review list, not an inferred complete call graph. Inheritance includes Factorial→Reactivation→Budget→Local. Full writer coverage must also exclude external direct writes, monkeypatching, unknown imports, callbacks, dynamic dispatch and hidden aliases. The present map cannot prove that exclusion. No restore/load/setstate-like calls were observed by the registered lexical patterns in these five files; that does not establish absence of restore routes elsewhere. Such routes remain quarantined until separately source-bound and assigned a new epoch/verified ownership path.

## Alias and source evidence

Each function record contains source path/hash, symbol/line range, excerpt hash, possible alias names, syntax events and its proposed integration disposition. The scanner follows assignments from self/model and propagates provenance through local aliases and method/copy results; table writes through `count` therefore remain visible. It deliberately ignores assignment order and may overreport copied values, tuple-unpacking outputs and independent primitive data. This is not Python points-to analysis. Nested scopes, imported code and runtime mutation are not resolved or executed.

Observed syntax events: 108 rooted stores, 10 rooted mutator calls, 141 rooted alias assignments, 252 rooted argument transfers, 27 rooted returns, 19 outcome attributes, five outcome/labels parameters and zero restore-like calls. Events overlap and include host-control and generator state; they are not counts of learning updates, unique mutable aliases, actual escapes or observed leaks. A function with no detected write is not certified pure. The complete 62-function list is complete only for the selected syntactic definition scope, never for all program execution paths.

## Input/outcome separation

Current training rows are outcome-bearing Samples; the runner can access `row.outcome` before observe. Reactivation train also receives a labels sequence. Calling begin before observe in a loop is not a channel boundary that makes the result unavailable. Predictions, losses, evaluator true dependencies and outcome shuffle maps must not be passed into an outcome-free collector.

A new runner needs an input-only event envelope with owner/epoch/receiver and source address, no result payload. A separate result authority may release the own delayed outcome only after capture acknowledgement binds both stable predecessor and pending receipt revisions. The collector can acknowledge a complete frozen capture but cannot author its own owner/version or use a source address as its clock. Missing capture, rejected acknowledgement, stale generation or exception must leave release blocked and stop processing rather than supplying labels early. No such channel or real data producer is implemented here.

## Verification, required work and holds

Six tests check finite inventory without importing audited modules, local count-alias writes, inherited and host method presence, conservative copy provenance, exact repeated nonmutation, and source/report/roster/budget refusal. Required-source checks bind pending stores, scheduler list mutation, table allocation/alias update, raw model handoff and outcome access. Fresh-process primary/replay match; registered sources and all 629 historical hashes verify. Consumed primary/replay refuse before computation. The separate four-gate roadmap check remains valid.

Required changes remain: private host construction with no raw model export; complete writer/diagnostic gating; distinct stable pre-begin/post-begin phases; input-only producer with acknowledged delayed outcome; independent primitive copying without calculation helpers; and quarantine of restore/copy/evaluator paths. Stop for any unresolved alias/dynamic writer, unreviewed inherited dispatch, early outcome visibility, copy/restore bypass, consumed/sealed-run dependency or exceeded budget. A new implementation protocol must preserve existing arithmetic, exact actual-update conditions and original acceptance criteria, not reinterpret this map as a training authorization.

🟠 [Next] Preregister a model-free input/outcome admission interface: input-only envelopes, owner/epoch and receiver/source identity, capture-acknowledgement binding and delayed-result release/refusal. Validate absence of result payload before acknowledgement and fail-closed stale/duplicate/partial acknowledgement handling without model/calculator/storage calls. This addresses the specific source-observed outcome-firewall gap; it does not attach a host or establish real producer provenance.

⭕️ [Pending] Live host integration, genuine owner/version authority, complete writer/alias confinement, actual outcome-free capture, independent scalar execution, durable nine-route atomicity, independent task usefulness and SNN-specific causal evidence. All seven missing categories remain missing. No new scientific trial, score, acquisition, resource measurement, condition adjustment or seal opening occurred. Original reproduction failure and recovery/retention/resource negatives remain. Generic ownership and channel separation are equally applicable to scalar systems, not evidence of an SNN-exclusive advantage.

## 日本語

5ソースの62関数・メソッドを固定し、更新・局所エイリアス・参照の受け渡し・継承ディスパッチ・結果アクセスを静的に整理しました。6テストと別プロセス再検証が成功し、629件の既存証跡を維持しました。count経由のテーブル更新、スケジューラーや診断状態、コピーと評価経路も管理対象です。全プログラムの更新経路保証やエイリアス封じ込めの証明ではありません。入力と結果を別チャネルにし、採取確認前の結果配送を止める変更が必要です。学習器・計算器・モデル保存処理・ホスト・実接続は実行せず、旧失敗・負の結果・封印を維持しました。

## 简体中文

固定5个源文件中的62个函数/方法，静态整理写入、局部别名、引用传递、继承派发和结果访问。6项测试与独立进程复验通过，保留629项历史证据。count别名的表更新、调度器/诊断状态、复制及评估路径均列入管理范围。这不是全部运行路径覆盖或别名隔离的证明。需要分离输入/结果通道，在采集确认前阻止结果配送。不执行学习器、计算器、模型存储、主机或实际连接，旧失败、负面结果与封存保持不变。
