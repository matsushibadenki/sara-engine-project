# SARA Engine Roadmap

This page tracks current progress, priorities and the conditions for advancing. The historical roadmap contains the full experiment records, earlier decisions and detailed phase plans. [Roadmap history](ROADMAP_HISTORY.md)

## Current progress

| Status | Area | Evidence and remaining work |
| --- | --- | --- |
| 🟢 [Done] | Local route learning and capture | Actual local updates, delayed outcomes, retained-key probabilities and receipt acknowledgements verified in bounded synthetic runs. [Factorization evidence](SPARSE_PREDICTIVE_FACTORIZATION.md) |
| 🟢 [Done] | Independent scalar comparison | 700 receipts and 6,291 local updates matched through two reactivation cycles and their cooldowns. [Comparison](SPARSE_ROUTE_MULTICYCLE_COMPARISON.md) |
| 🟢 [Done] | Durable receipt processing | Nine-route updates and receipt consumption commit together. Sequential processing, duplicate refusal and recovery were verified with local SQLite. [Durability](SPARSE_ROUTE_DURABLE_COMMIT.md) |
| 🟢 [Done] | Authenticated delivery | Separate test producers send authenticated JSON to the stdin CLI. Key rotation, tamper rejection and idempotent replay across restarts pass. [Ingress](SPARSE_ROUTE_DURABLE_COMMIT.md#separate-process-delivery-ingress) |
| 🟢 [Done] | Pilot observation summary | Client reports aggregate refresh acceptance, answer rates, measured request/body-byte totals and maximum refresh latency. Missing measurements remain explicit; independent acceptance remains pending. [Measurement scope](ROADMAP_NEXT_READINESS.md#pilot-observation-summary) |
| ⭕️ [Pending] | Independent task and SNN effect | Qualify an independent source and test the causal contribution of the SNN mechanism against matched event-driven scalar controls. [Decision gate](R2_MECHANISM_DECISION_GATE.md) |
| ⭕️ [Pending] | Operational deployment | Independent producers, reviewed key distribution and broader writer coverage are still required. Candidate V2 launch remains disabled. [Capture candidate](SPARSE_ROUTE_CAPTURE_CANDIDATE_V2.md) |

## Priorities

1. 🟠 [Next] Identify a fresh independently observed delayed-outcome task. Require repeated current states, distinguishable histories, both classes and the existing support and source gates before preregistering a comparison. Use the strongest current-state control and match actual local writes. [KNMI Schiphol provenance and split review](ROADMAP_NEXT_READINESS.md#schiphol-provenance-and-split-follow-up) · [KSEA 112-byte format follow-up](R2_ASOS_KSEA_112_BYTE_FORMAT_REVIEW.md#legacy-documentation-and-published-examples-follow-up) · [KSEA transport V2 framing stop](R2_ASOS_KSEA_TRANSPORT_V2.md#execution-result) · [KSEA format audit and acquisition stop](R2_ASOS_KSEA_FORMAT_AUDIT.md) · [KSEA first-sensor normalization](R2_ASOS_KSEA_FIRST_SENSOR_NORMALIZATION.md) · [KSEA delayed outcome contract](R2_ASOS_KSEA_DELAYED_CONTRACT.md) · [KSEA development timestamp preflight](R2_ASOS_KSEA_TIMESTAMP_PREFLIGHT.md) · [ASOS KSEA inventory and visibility specification](ROADMAP_NEXT_READINESS.md#asos-ksea-inventory-and-visibility-specification) · [ASOS independent task screen](ROADMAP_NEXT_READINESS.md#asos-independent-task-metadata-screen) · [IEBB appendix follow-up](ROADMAP_NEXT_READINESS.md#iebb-appendix-follow-up) · [Overijssel IEBB dwelling screen](ROADMAP_NEXT_READINESS.md#overijssel-iebb-dwelling-screen) · [Minneapolis homedata release screen](ROADMAP_NEXT_READINESS.md#minneapolis-homedata-release-screen) · [Fritz and Cook provenance follow-up](ROADMAP_NEXT_READINESS.md#fritz-and-cook-supplement-and-provenance-follow-up) · [Longitudinal ambient sensing and ARAS screen](ROADMAP_NEXT_READINESS.md#longitudinal-ambient-sensing-and-aras-screen) · [TigerPlace deployment follow-up](ROADMAP_NEXT_READINESS.md#tigerplace-deployment-follow-up) · [TigerPlace and Tokyo smart-room screen](ROADMAP_NEXT_READINESS.md#tigerplace-and-tokyo-smart-room-screen) · [TIHM release follow-up](ROADMAP_NEXT_READINESS.md#tihm-release-follow-up) · [CART dictionary follow-up and TIHM screen](ROADMAP_NEXT_READINESS.md#cart-dictionary-follow-up-and-tihm-screen) · [ORCATECH and Orange4Home screen](ROADMAP_NEXT_READINESS.md#orcatech-and-orange4home-screen) · [E-care@home supplement follow-up](ROADMAP_NEXT_READINESS.md#e-carehome-supplement-follow-up) · [Native-timestamp entrance sensor screen](ROADMAP_NEXT_READINESS.md#native-timestamp-entrance-sensor-screen) · [HOMESENSE release-code audit](ROADMAP_NEXT_READINESS.md#homesense-release-code-audit) · [SHAPES follow-up and HOMESENSE priority](ROADMAP_NEXT_READINESS.md#shapes-follow-up-and-homesense-priority) · [Household uptime screen](ROADMAP_NEXT_READINESS.md#independent-source-screen-with-household-uptime) · [Four-country coverage follow-up](ROADMAP_NEXT_READINESS.md#four-country-coverage-follow-up) · [Alternative-source screen](ROADMAP_NEXT_READINESS.md#refit-alternatives--2026-11-14) · [REFIT metadata review](ROADMAP_NEXT_READINESS.md#independent-delayed-outcome-task) · [Independent-task history and gates](ROADMAP_HISTORY.md#existing-research-queue)
2. 🟠 [Next] Prepare the independent producer pilot with reviewed trust material, revision handling and key distribution. Measure authority review, revision completeness, availability, latency and complete-path cost under the existing pilot conditions. [Live pilot acceptance procedure](ROADMAP_NEXT_READINESS.md#live-pilot-acceptance-procedure) · [Publisher pilot readiness](ROADMAP_NEXT_READINESS.md#independent-https-publisher-pilot) · [Publisher pilot context](ROADMAP_HISTORY.md#existing-research-queue)
3. ⭕️ [Pending] Promote a mechanism or begin new resource comparisons only after the independent task and causal gates pass. Existing event-work and allocation diagnostics retain their recorded results. [Research focus and promotion gates](ROADMAP_NEXT_LEVEL.md)
4. 🔴 [Later] Compare bounded local representations at equal total byte budgets, including indexes, decoding, residuals and rare-route retention. [Early route selection](#early-route-selection) · [LittleBit design reference](SPARSE_PREDICTIVE_FACTORIZATION.md#littlebit-design-reference)
5. 🔴 [Later] Evaluate exact reuse of local readouts when their dependencies are unchanged, after the independent-task and causal gates pass. [Evaluation contract](#local-readout-reuse)

6. 🔴 [Later] Evaluate bounded inference escalation after the same independent-task and causal gates pass. [Evaluation contract](#bounded-inference-escalation)
7. 🔴 [Later] Evaluate local repair with explicit constraints and a bounded stopping rule. [Evaluation design](#local-constraint-repair)
8. 🔴 [Later] Compare targeted local-state edits on isolated copies. [Evaluation design](#local-state-edit-evaluation)

### Early route selection

The Lottery Ticket Hypothesis motivates controlling separately for the retained structure and its initial state. In SARA, test whether a bounded early development prefix can identify local routes worth retaining, using only evidence available at the checkpoint. Take the checkpoint after pending outcomes have settled within the registered horizon. Preserve rare routes with insufficient support and evaluate masking before committing any structural deletion. [Frankle and Carbin](https://arxiv.org/abs/1803.03635)

At the same retained byte budget, compare selected routes with retained local state, the same selection with reset local state, a seeded random selection at the same checkpoint, and a sparse-from-start arm. Keep the unpruned run as a quality reference. Use identical follow-up receipts and delayed outcomes. Fix the update allocation in advance and verify actual local writes for the mechanism comparison. If workloads differ in an end-to-end saving comparison, report the write difference separately. Give the event-driven scalar control the same selection opportunity. Measure immediate loss separately from recovery and dependency retention, including rare histories and regime shifts. Include selection, temporary branches, indexes and fallback in total cost.

日本語：初期の開発データだけで残す経路を選び、経路選択と局所状態の保持の効果を分けて比較する将来候補を追加。

简体中文：新增后续候选，仅在开发期早段选择保留路径，分别检验路径选择与局部状态保留的效果。

### Local readout reuse

The referenced computation-reduction proposal motivates reusing a derived local readout while its key, contributing state versions and time-dependent inputs remain equal. Start with a bounded per-route cache for a read-only probability. Recompute on a missing entry, dependency change or uncertain validity. An unchanged input can still coincide with trace decay, a delayed result, expiry or a restored state, so these transitions must be covered by the reuse check. Capture the probability at receipt time and preserve it through delivery.

Require exact output and state parity with the uncached implementation, including delayed results, expiry, recovery and rare-route cases. Apply the same reuse opportunity to the event-driven scalar control and match actual local writes. Account for dependency checks, cache memory, invalidation, fallback and shared event processing using the existing [work accounting contract](R2_EVENT_WORK_ACCOUNTING.md). Attribute a benefit to SNN dynamics only if an advantage survives the matched scalar comparison and the existing causal gate.

Treat error-based omission of local writes as a separate learning intervention. Small errors alone provide insufficient evidence to skip updates that may preserve rare dependencies or aid recovery. Semantic planning and text rendering separation remains outside this route-level comparison.

日本語：依存状態が一致する局所読み出しの再利用を将来候補に追加。遅延結果・期限切れ・復旧を含む完全一致と、同じ再利用を備えたスカラー対照を条件とする。

简体中文：将依赖状态一致时复用局部读出列为后续候选，要求在延迟结果、过期和恢复时完全一致，并采用具备相同复用机制的标量对照。

### Bounded inference escalation

Start with the existing current-state readout. Use development-only evidence of support and uncertainty to decide whether to consult the temporal readout. Escalate on unknown keys or insufficient support. Fix the policy and quality margins before evaluation. Inference routing leaves the receipt lifecycle and local write schedule intact.

Compare always-current, always-temporal and conditional readouts, and give the event-driven scalar control the same routing opportunity. Check aggregate quality and the quality of predictions that stayed on the cheap path, including rare histories and delayed outcomes. Use the existing work-accounting contract to include routing, both readouts on fallback, shared event processing and memory. Report fallback frequency and tail latency alongside total work. Evaluate this policy separately from exact reuse before testing their combination.

日本語：現在状態から読み出し、根拠が不足する場合に時間履歴を参照する段階推論を将来候補に追加。

简体中文：将分级推理列为后续候选，先读取当前状态，证据不足时再读取时间历史。

### Local constraint repair

CRAM motivates updating one variable at a time, scoring a change from its incident constraints, and checking whether further search is needed. Explore this on a small CPU-only constraint task with fixed observations. Maintain an index from variables to constraints and update only affected scores. Learn any weights from local training evidence without differentiating through the search. Compare learned weights with hand-set weights and a direct constraint solver. [Hoover, Blayney and Whittington, arXiv v1](https://arxiv.org/html/2611.17309v1)

Accept early only when an independent verifier confirms the required constraints. For tasks with soft constraints, register the quality requirement separately. Keep a fixed step budget and record exhaustion or stalled search. Compare against fixed-budget search and give the event-driven scalar control the same stopping rule. Count score updates, queue operations and verification work at matched solution quality. The paper's GPU timings and puzzle results apply to its own benchmarks. Use the existing promotion gates before new resource comparisons or claims of SNN-specific benefit.

日本語：局所的な制約修復と検証付き早期停止を将来候補に追加。手設計と直接ソルバーも対照に含める。

简体中文：将局部约束修复和经验证的提前停止列为后续候选，并与手工设置及直接求解器比较。

### Local-state edit evaluation

Adapt Fan et al.'s paired edit evaluation to a bounded copy of SARA's local route state. Use development diagnostics to select a fixed set of fields, apply the proposed edit to a copy, and validate the exact write set and resulting numerical deltas. Keep the original snapshot available for recovery. Treat recovery from injected damage and improvement of intact state as separate hypotheses. [Fan et al., arXiv v1](https://arxiv.org/html/2611.21885v1)

Compare targeted edits with a no-edit reference and a seeded control that changes the same number of local fields by comparable magnitudes. Fix the evaluation protocol before proposing edits and use paired inputs and delayed outcomes. Check target recovery and unrelated-route retention, including rare histories. Record withdrawals and regressions. Candidate selection uses development evidence, with sealed evaluation subject to the existing gates.

日本語：局所状態の複製に対する編集を対応比較し、故障からの回復と正常状態の改善を別々に検証する。

简体中文：在局部状态副本上配对比较编辑效果，分别验证故障恢复与完好状态的改进。

### Reconstruct local arithmetic from a written specification

Adopt the iWSP reconstruction audit as a research and implementation target for SARA's local readout. The paper tests whether a fresh session can rebuild a model from a short description. SARA can apply a stricter, numerical check to its deterministic local arithmetic. [Mehta et al., arXiv v1](https://arxiv.org/html/2611.18114v1)

The specification defines allocation, prior mass, probability and the receipt-time error increment. Give only the specification and input interface to an independent implementer. Freeze the reconstructed implementation before comparing it against the existing eleven literal records. Require exact agreement for probability and error increment, including the distinction between an unallocated key and allocated zero counts. Add separately registered boundary cases to distinguish rules that happen to agree on the existing examples. Keep the original expected values and test seals intact. A pass would establish reproducibility of the documented rules on those cases. [Existing independent arithmetic test](../tests/test_route_independent_scalar_preflight.py)

🟢 [Done] The written rules and boundary expectations were registered in commit `e714ec3`. The standalone implementation was committed as `0f1d8be` before running the comparison. All eleven literal records and four registered boundary cases matched exactly for both outputs. This was a separate implementation with no imports from the existing arithmetic, developed in the same session as the specification. [Registered specification](LOCAL_ARITHMETIC_RECONSTRUCTION.md)

🟢 [Done] A separate agent received only the registered calculation rules and interface, with no inherited conversation or tool use. Its implementation was frozen in `00c46bf` before comparison. Probability and receipt-time error increment matched exactly on all 11 literal records and four registered boundaries. The unallocated and allocated-zero cases retained distinct probabilities of null and 0.5 with equal error increments. The comparison used only registered hand records.

🟢 [Done] The registered rounding audit found zero probability mismatches and eight increment mismatches across 8,192 cases. Multiplication matched the rational oracle in those eight cases. Specify the squaring operator explicitly in any future bitwise contract. [Audit and first counterexample](LOCAL_ARITHMETIC_RECONSTRUCTION.md#rounding-audit-results)

🟢 [Done] Specification v2 fixes the square to binary64 multiplication and defines the rounding order. The versioned implementation matched the rational reference for both outputs on all 8,207 cases. The eleven literal records and four boundary cases also retained exact agreement with their registered expectations. [Specification v2](LOCAL_ARITHMETIC_RECONSTRUCTION.md#local-arithmetic-specification-v2)

🟢 [Done] A separate agent received only the v2 calculation rules and interface, without inherited context, repository access or expected values. Its code was frozen in `bb8df9b` before testing. The blind reconstruction matched both rational-reference outputs on all 8,207 cases, including the unallocated and allocated-zero boundaries. The audit used the existing 11 literal records, four boundaries and 8,192 generated cases.

🟢 [Done] The v2 transition and count-limit audit was registered in `3c3fa46`. Both frozen v2 implementations matched the rational reference for both outputs on all 4,352 cases per implementation. Coverage included every tick scale transition, maximum and near-maximum masses, and all history, weighting and outcome combinations. The audit used generated inputs and kept the historical evidence and test seals intact. [Registered cases](LOCAL_ARITHMETIC_RECONSTRUCTION.md#v2-limit-and-transition-audit-preregistration)

🟢 [Done] Both frozen v2 implementations rejected all 18 registered invalid inputs with `ValueError`. The cases covered out-of-range ticks and counts, booleans in integer fields, unknown rules and outcomes, and nonzero masses on unallocated keys. Both zero-mass controls still matched the rational reference. [Registered invalid input cases](LOCAL_ARITHMETIC_RECONSTRUCTION.md#v2-invalid-input-audit-preregistration)

🟢 [Done] Each frozen v2 implementation rejected all 32 registered invalid types with `ValueError`. The audit covered floats, numeric strings and nonfinite numbers in integer fields, non-booleans in allocation, and non-strings in history, weighting and outcome. [Registered type cases](LOCAL_ARITHMETIC_RECONSTRUCTION.md#v2-invalid-type-audit-preregistration)

🟢 [Done] Both frozen v2 implementations rejected all 13 registered subclass and conversion-probe cases with `ValueError`. No instrumented conversion or comparison method ran. [Registered audit](LOCAL_ARITHMETIC_RECONSTRUCTION.md#v2-subclass-and-coercion-audit-preregistration)

🟢 [Done] Both frozen v2 implementations matched the rational reference for both outputs on 32,768 generated cases each. The audit covered every tick, asymmetric small and maximum counts, and all history, weighting and outcome combinations for those count pairs. [Registered cases](LOCAL_ARITHMETIC_RECONSTRUCTION.md#v2-all-tick-audit-preregistration)

日本語　短い仕様だけから局所計算を再実装し、固定記録で数値と欠損の意味を照合する。

简体中文　仅依据简短规格重建局部计算，并用固定记录核对数值与缺失状态。

## Selected execution gates

- 🟢 [Done] Fixed CartPole comparison completed with a negative V2 gate; no retry. <!-- gate:cartpole-fixed-comparison -->
- ⭕️ [Pending] New task/hypothesis and fresh development identities before another local-credit trial. <!-- gate:cartpole-fresh-task -->
- ⭕️ [Pending] Trusted owner/revision/pause and outcome-free access before actual receipt capture. <!-- gate:authoritative-receipt-capture -->
- ⭕️ [Pending] SNN-specific causal benefit over matched event-driven scalar controls. <!-- gate:snn-specific-advantage -->

## Evidence and phase plans

- [Roadmap history through the delivery ingress milestone](ROADMAP_HISTORY.md) preserves the previous roadmap, including negative and failed attempts, execution decisions and Phases 6–20.
- [Implemented features](IMPLEMENTED_FEATURES.md) lists completed capabilities. [Pre-v1.1 archive](old/ROADMAP_v1_1_completion_archive.md) records the release baseline.
- [Next-level roadmap](ROADMAP_NEXT_LEVEL.md) tracks promotion conditions and links to the detailed plans for Phases 21–46.
- Consumed experiments and sealed splits retain their recorded status. The selected execution gates are checked by the [roadmap gate checker](ROADMAP_GATE_CHECK.md).

日本語：合成入力での局所学習、独立スカラーとの照合、復旧可能な永続化、認証付き受信まで完了。次は独立課題の選定と外部送信元の試験準備。SNN固有の効果と資源上の優位性は検証待ち。

简体中文：已完成合成输入下的局部学习、独立标量对照、可恢复持久化与认证接收。下一步是筛选独立任务和准备外部发布方试点。SNN 特有的因果效果和资源优势仍待验证。

## Release baseline audit

* **ROADMAP closure audit:**
  - DONE: release-critical path is complete for the v1.1 baseline.
  - DONE: observed-only evidence remains labeled until stronger external or physical measurements pass.
  - DONE: long-term research backlog is separated into Phase 6 and later roadmap work.
  - DONE: roadmap completion audit is present for the current baseline.
  - DONE: research product completion gate is part of the active release evidence surface.
